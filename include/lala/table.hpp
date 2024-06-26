// Copyright 2023 Pierre Talbot

#ifndef LALA_POWER_TABLES_HPP
#define LALA_POWER_TABLES_HPP

#include "battery/vector.hpp"
#include "battery/shared_ptr.hpp"
#include "battery/dynamic_bitset.hpp"
#include "lala/logic/logic.hpp"
#include "lala/universes/primitive_upset.hpp"
#include "lala/abstract_deps.hpp"

namespace lala {

template <class A, class U, class Alloc> class Table;
namespace impl {
  template <class>
  struct is_table_like {
    static constexpr bool value = false;
  };
  template<class A, class U, class Alloc>
  struct is_table_like<Table<A, U, Alloc>> {
    static constexpr bool value = true;
  };
}

/** The table abstract domain is designed to represent predicates in extension by listing all their solutions explicitly.
 * It is inspired by the table global constraint and generalizes it by lifting each element of the table to a lattice element.
 * We expect `U` to be equally or less expressive than `A::universe_type`, this is because we compute the meet in `A::universe_type` and not in `U`.
 */
template <class A, class U = typename A::universe_type, class Allocator = typename A::allocator_type>
class Table {
public:
  using allocator_type = Allocator;
  using sub_allocator_type = typename A::allocator_type;
  using universe_type = U;
  using local_universe = typename universe_type::local_type;
  using sub_universe_type = typename A::universe_type;
  using sub_local_universe = typename sub_universe_type::local_type;
  using memory_type = typename universe_type::memory_type;
  using sub_type = A;
  using sub_ptr = abstract_ptr<sub_type>;
  using this_type = Table<sub_type, universe_type, allocator_type>;

  constexpr static const bool is_abstract_universe = false;
  constexpr static const bool sequential = sub_type::sequential;
  constexpr static const bool is_totally_ordered = false;
  constexpr static const bool preserve_bot = sub_type::preserve_bot;
  constexpr static const bool preserve_top = sub_type::preserve_top;
  // The next properties should be checked more seriously, relying on the sub-domain might be uneccessarily restrictive.
  constexpr static const bool preserve_join = sub_type::preserve_join;
  constexpr static const bool preserve_meet = sub_type::preserve_meet;
  constexpr static const bool injective_concretization = sub_type::injective_concretization;
  constexpr static const bool preserve_concrete_covers = sub_type::preserve_concrete_covers;
  constexpr static const char* name = "Table";

  using table_type = battery::vector<universe_type, allocator_type>;
  using bitset_type = battery::dynamic_bitset<memory_type, allocator_type>;

private:
  AType atype;
  AType store_aty;
  sub_ptr sub;

  // For each instance `i` of the table, we have its set of variables `headers[i]`.
  battery::vector<battery::vector<AVar, allocator_type>, allocator_type> headers;
  table_type tell_table;
  table_type ask_table;
  battery::vector<bitset_type, allocator_type> eliminated_rows;

  // We keep a bitset representation of each variable in the table.
  // We perform a reduced product between this representation and the underlying domain.
  battery::vector<bitset_type, allocator_type> bitset_store;

public:
  template <class Alloc>
  struct tell_type {
    using allocator_type = Alloc;

    typename A::template tell_type<Alloc> sub;

    battery::vector<battery::vector<AVar, Alloc>, Alloc> headers;

    battery::vector<universe_type, Alloc> tell_table;
    battery::vector<universe_type, Alloc> ask_table;

    CUDA tell_type(const Alloc& alloc = Alloc{})
     : sub(alloc)
     , headers(alloc)
     , tell_table(alloc)
     , ask_table(alloc)
    {}
    tell_type(const tell_type&) = default;
    tell_type(tell_type&&) = default;
    tell_type& operator=(tell_type&&) = default;
    tell_type& operator=(const tell_type&) = default;

    template <class TableTellType>
    CUDA NI tell_type(const TableTellType& other, const Alloc& alloc = Alloc{})
      : sub(other.sub, alloc)
      , headers(other.headers, alloc)
      , tell_table(other.tell_table, alloc)
      , ask_table(other.ask_table, alloc)
    {}

    CUDA allocator_type get_allocator() const {
      return headers.get_allocator();
    }

    template <class Alloc2>
    friend class tell_type;
  };

  template <class Alloc>
  struct ask_type {
    using allocator_type = Alloc;

    typename A::template ask_type<Alloc> sub;
    battery::vector<battery::vector<AVar, Alloc>, Alloc> headers;
    battery::vector<universe_type, Alloc> ask_table;

    CUDA ask_type(const Alloc& alloc = Alloc{})
     : sub(alloc)
     , headers(alloc)
     , ask_table(alloc)
    {}
    ask_type(const ask_type&) = default;
    ask_type(ask_type&&) = default;
    ask_type& operator=(ask_type&&) = default;
    ask_type& operator=(const ask_type&) = default;

    template <class TableAskType>
    CUDA NI ask_type(const TableAskType& other, const Alloc& alloc = Alloc{})
      : sub(other.sub, alloc)
      , headers(other.headers, alloc)
      , ask_table(other.ask_table, alloc)
    {}

    CUDA allocator_type get_allocator() const {
      return headers.get_allocator();
    }

    template <class Alloc2>
    friend class ask_type;
  };

  template <class A2, class U2, class Alloc2>
  friend class Table;

public:
  CUDA Table(AType uid, AType store_aty, sub_ptr sub, const allocator_type& alloc = allocator_type())
   : atype(uid)
   , store_aty(store_aty)
   , sub(std::move(sub))
   , headers(alloc)
   , tell_table(alloc)
   , ask_table(alloc)
   , eliminated_rows(alloc)
  {}

  CUDA Table(AType uid, sub_ptr sub, const allocator_type& alloc = allocator_type())
   : Table(uid, sub->aty(), sub, alloc)
  {}

  template<class A2, class U2, class Alloc2, class... Allocators>
  CUDA NI Table(const Table<A2, U2, Alloc2>& other, AbstractDeps<Allocators...>& deps)
   : atype(other.atype)
   , store_aty(other.store_aty)
   , sub(deps.template clone<sub_type>(other.sub))
   , headers(other.headers, deps.template get_allocator<allocator_type>())
   , tell_table(other.tell_table, deps.template get_allocator<allocator_type>())
   , ask_table(other.ask_table, deps.template get_allocator<allocator_type>())
   , eliminated_rows(other.eliminated_rows, deps.template get_allocator<allocator_type>())
  {}

  CUDA AType aty() const {
    return atype;
  }

  CUDA allocator_type get_allocator() const {
    return headers.get_allocator();
  }

  CUDA sub_ptr subdomain() const {
    return sub;
  }

  CUDA local::BDec is_bot() const {
    return tell_table.size() == 0 && sub->is_bot();
  }

  CUDA local::BInc is_top() const {
    for(int i = 0; i < eliminated_rows.size(); ++i) {
      if(eliminated_rows[i].count() == tell_table.size()) {
        return true;
      }
    }
    return sub->is_top();
  }

  CUDA static this_type bot(AType atype = UNTYPED,
    AType atype_sub = UNTYPED,
    const allocator_type& alloc = allocator_type(),
    const sub_allocator_type& sub_alloc = sub_allocator_type())
  {
    return Table{atype, battery::allocate_shared<sub_type>(alloc, sub_type::bot(atype_sub, sub_alloc)), alloc};
  }

  /** A special symbolic element representing top. */
  CUDA static this_type top(AType atype = UNTYPED,
    AType atype_sub = UNTYPED,
    const allocator_type& alloc = allocator_type(),
    const sub_allocator_type& sub_alloc = sub_allocator_type())
  {
    return Table{atype, battery::allocate_shared<sub_type>(sub_alloc, sub_type::top(atype_sub, sub_alloc)), alloc};
  }

  template <class Env>
  CUDA static this_type bot(Env& env,
    const allocator_type& alloc = allocator_type(),
    const sub_allocator_type& sub_alloc = sub_allocator_type())
  {
    AType atype_sub = env.extends_abstract_dom();
    AType atype = env.extends_abstract_dom();
    return bot(atype, atype_sub, alloc, sub_alloc);
  }

  template <class Env>
  CUDA static this_type top(Env& env,
    const allocator_type& alloc = allocator_type(),
    const sub_allocator_type& sub_alloc = sub_allocator_type())
  {
    AType atype_sub = env.extends_abstract_dom();
    AType atype = env.extends_abstract_dom();
    return top(atype, atype_sub, alloc, sub_alloc);
  }

  template <class Alloc2>
  struct snapshot_type {
    using sub_snap_type = sub_type::template snapshot_type<Alloc2>;
    sub_snap_type sub_snap;
    size_t num_tables;

    snapshot_type(const snapshot_type<Alloc2>&) = default;
    snapshot_type(snapshot_type<Alloc2>&&) = default;
    snapshot_type<Alloc2>& operator=(snapshot_type<Alloc2>&&) = default;
    snapshot_type<Alloc2>& operator=(const snapshot_type<Alloc2>&) = default;

    template <class SnapshotType>
    CUDA snapshot_type(const SnapshotType& other, const Alloc2& alloc = Alloc2())
      : sub_snap(other.sub_snap, alloc)
      , num_tables(other.num_tables)
    {}

    CUDA snapshot_type(sub_snap_type&& sub_snap, size_t num_tables)
      : sub_snap(std::move(sub_snap))
      , num_tables(num_tables)
    {}
  };

  template <class Alloc2 = allocator_type>
  CUDA snapshot_type<Alloc2> snapshot(const Alloc2& alloc = Alloc2()) const {
    return snapshot_type<Alloc2>(sub->snapshot(alloc), headers.size());
  }

  template <class Alloc2>
  CUDA void restore(const snapshot_type<Alloc2>& snap) {
    sub->restore(snap.sub_snap);
    headers.resize(snap.num_tables);
    if(snap.num_tables == 0) {
      tell_table.resize(0);
      ask_table.resize(0);
    }
    eliminated_rows.resize(snap.num_tables);
    for(int i = 0; i < eliminated_rows.size(); ++i) {
      eliminated_rows[i].reset();
    }
  }

  template <class F>
  CUDA void flatten_and(const F& f, typename F::Sequence& conjuncts) const {
    if(f.is(F::Seq) && f.sig() == AND) {
      for(int i = 0; i < f.seq().size(); ++i) {
        flatten_and(f.seq(i), conjuncts);
      }
    }
    else {
      conjuncts.push_back(f);
    }
  }

  template <class F>
  CUDA void flatten_or(const F& f, typename F::Sequence& disjuncts) const {
    if(f.is(F::Seq) && f.sig() == OR) {
      for(int i = 0; i < f.seq().size(); ++i) {
        flatten_or(f.seq(i), disjuncts);
      }
    }
    else {
      typename F::Sequence conjuncts{disjuncts.get_allocator()};
      flatten_and(f, conjuncts);
      if(conjuncts.size() > 1) {
        disjuncts.push_back(F::make_nary(AND, std::move(conjuncts)));
      }
      else {
        disjuncts.push_back(std::move(conjuncts[0]));
      }
    }
  }

  template <class F>
  CUDA F flatten(const F& f, const typename F::allocator_type& alloc) const {
    typename F::Sequence disjuncts{alloc};
    flatten_or(f, disjuncts);
    if(disjuncts.size() > 1) {
      return F::make_nary(OR, std::move(disjuncts));
    }
    else {
      return std::move(disjuncts[0]);
    }
  }

  template <IKind kind, bool diagnose = false, class F, class Env, class Alloc>
  CUDA NI bool interpret_atom(
    battery::vector<AVar, Alloc>& header,
    battery::vector<battery::vector<local_universe, Alloc>, Alloc>& tell_table2,
    battery::vector<battery::vector<local_universe, Alloc>, Alloc>& ask_table2,
    const F& f, Env& env, IDiagnostics& diagnostics) const
  {
    if(num_vars(f) != 1) {
      RETURN_INTERPRETATION_ERROR("Only unary formulas are supported in the cell of the table.");
    }
    else {
      auto x_opt = var_in(f, env);
      if(!x_opt.has_value() || !x_opt->get().avar_of(store_aty).has_value()) {
        RETURN_INTERPRETATION_ERROR("Undeclared variable.");
      }
      AVar x = x_opt->get().avar_of(store_aty).value();
      int idx = 0;
      for(; idx < header.size() && header[idx] != x; ++idx) {}
      // If it's a new variable not present in the previous rows, we add it in each row with bottom value.
      if(idx == header.size()) {
        header.push_back(x);
        for(int i = 0; i < tell_table2.size(); ++i) {
          if constexpr(kind == IKind::TELL) {
            tell_table2[i].push_back(local_universe::bot());
          }
          ask_table2[i].push_back(local_universe::bot());
        }
      }
      local_universe ask_u{local_universe::bot()};
      if(ginterpret_in<IKind::ASK, diagnose>(f, env, ask_u, diagnostics)) {
        ask_table2.back()[idx].tell(ask_u);
        if constexpr(kind == IKind::TELL) {
          local_universe tell_u{local_universe::bot()};
          if(ginterpret_in<IKind::TELL, diagnose>(f, env, tell_u, diagnostics)) {
            tell_table2.back()[idx].tell(tell_u);
          }
          else {
            return false;
          }
        }
      }
      else {
        return false;
      }
    }
    return true;
  }

  template <class Alloc1, class Alloc2>
  CUDA battery::vector<local_universe, Alloc2> flatten_table(
    const battery::vector<battery::vector<local_universe, Alloc1>, Alloc1>& table, const Alloc2& alloc) const
  {
    if(table.size() == 0) { return battery::vector<local_universe, Alloc2>(alloc);}
    battery::vector<local_universe, Alloc2> res(alloc);
    res.reserve(table.size() * table[0].size());
    for(int i = 0; i < table.size(); ++i) {
      for(int j = 0; j < table[i].size(); ++j) {
        res.push_back(table[i][j]);
      }
    }
    return std::move(res);
  }

public:
  template <IKind kind, bool diagnose = false, class F, class Env, class I>
  CUDA NI bool interpret(const F& f2, Env& env, I& intermediate, IDiagnostics& diagnostics) const {
    F f = flatten(f2, env.get_allocator());
    using Alloc = typename I::allocator_type;
    if(f.is(F::Seq) && f.sig() == OR) {
      battery::vector<AVar, Alloc> header(intermediate.get_allocator());
      battery::vector<battery::vector<local_universe, Alloc>, Alloc> tell_table2(intermediate.get_allocator());
      battery::vector<battery::vector<local_universe, Alloc>, Alloc> ask_table2(intermediate.get_allocator());
      for(int i = 0; i < f.seq().size(); ++i) {
        // Add a row in the table.
        tell_table2.push_back(battery::vector<local_universe, Alloc>(header.size(), local_universe::bot(), intermediate.get_allocator()));
        ask_table2.push_back(battery::vector<local_universe, Alloc>(header.size(), local_universe::bot(), intermediate.get_allocator()));
        if(f.seq(i).is(F::Seq) && f.seq(i).sig() == AND) {
          const auto& row = f.seq(i).seq();
          for(int j = 0; j < row.size(); ++j) {
            size_t error_ctx = diagnostics.num_suberrors();
            if(!interpret_atom<kind, diagnose>(header, tell_table2, ask_table2, row[j], env, diagnostics)) {
              if(!sub->template interpret<kind, diagnose>(f2, env, intermediate.sub, diagnostics)) {
                return false;
              }
              diagnostics.cut(error_ctx);
              return true;
            }
          }
        }
        else {
          size_t error_ctx = diagnostics.num_suberrors();
          if(!interpret_atom<kind, diagnose>(header, tell_table2, ask_table2, f.seq(i), env, diagnostics)) {
            if(!sub->template interpret<kind, diagnose>(f2, env, intermediate.sub, diagnostics)) {
              return false;
            }
            diagnostics.cut(error_ctx);
            return true;
          }
        }
      }
      // When only one variable is present, we leave its interpretation to the subdomain.
      if(header.size() > 1) {
        auto ask_table2_flatten = flatten_table(ask_table2, ask_table2.get_allocator());
        if((intermediate.ask_table.size() == 0 || intermediate.ask_table == ask_table2_flatten) &&
          (ask_table.size() == 0 || ask_table == ask_table2_flatten))
        {
          intermediate.headers.push_back(std::move(header));
          if(intermediate.ask_table.size() == 0 && ask_table.size() == 0) {
            if constexpr(kind == IKind::TELL) {
              intermediate.tell_table = flatten_table(tell_table2, tell_table2.get_allocator());
            }
            intermediate.ask_table = std::move(ask_table2_flatten);
          }
          return true;
        }
        else {
          RETURN_INTERPRETATION_ERROR("This abstract domain can only represent multiple tables over a same matrix of values, but a difference was detected.");
        }
      }
    }
    return sub->template interpret<kind, diagnose>(f, env, intermediate.sub, diagnostics);
  }

  template <bool diagnose = false, class F, class Env, class Alloc>
  CUDA NI bool interpret_ask(const F& f, const Env& env, ask_type<Alloc>& ask, IDiagnostics& diagnostics) const {
    return interpret<IKind::ASK, diagnose>(f, const_cast<Env&>(env), ask, diagnostics);
  }

  template <bool diagnose = false, class F, class Env, class Alloc>
  CUDA NI bool interpret_tell(const F& f, Env& env, tell_type<Alloc>& tell, IDiagnostics& diagnostics) const {
    return interpret<IKind::TELL, diagnose>(f, env, tell, diagnostics);
  }

  CUDA const sub_universe_type& operator[](int x) const {
    return (*sub)[x];
  }

  CUDA size_t vars() const {
    return sub->vars();
  }

  CUDA size_t num_tables() const {
    return headers.size();
  }

private:
  template <IKind kind>
  CUDA sub_local_universe convert(const local_universe& x) const {
    if constexpr(std::is_same_v<universe_type, sub_universe_type>) {
      return x;
    }
    else {
      VarEnv<battery::standard_allocator> env;
      IDiagnostics diagnostics;
      sub_local_universe v{sub_local_universe::bot()};
      bool succeed = ginterpret_in<kind>(x.deinterpret(AVar{}, env), env, v, diagnostics);
      assert(succeed);
      return v;
    }
  }

  CUDA size_t num_columns() const {
    return headers[0].size();
  }

  CUDA size_t num_rows() const {
    return tell_table.size() / num_columns();
  }

public:
  template <class Alloc, class Mem>
  CUDA this_type& tell(const tell_type<Alloc>& t, BInc<Mem>& has_changed) {
    if(t.headers.size() > 0) {
      has_changed.tell_top();
    }
    sub->tell(t.sub, has_changed);
    // If there is a table in the tell, we add it to the current abstract element.
    if(t.tell_table.size() > 0) {
      // Since this abstract element only handle one matrix of elements at a time, the current table must be empty.
      assert(tell_table.size() == 0);
      tell_table = t.tell_table;
      ask_table = t.ask_table;
    }
    // For each new table (sharing the matrix of elements).
    for(int i = 0; i < t.headers.size(); ++i) {
      // Each table must have the same number of columns.
      headers.push_back(battery::vector<AVar, allocator_type>(t.headers[i], get_allocator()));
      eliminated_rows.push_back(bitset_type(num_rows(), get_allocator()));
    }
    return *this;
  }

  template <class Alloc>
  CUDA this_type& tell(const tell_type<Alloc>& t)  {
    local::BInc has_changed;
    return tell(t, has_changed);
  }

  CUDA this_type& tell(AVar x, const sub_universe_type& dom) {
    sub->tell(x, dom);
    return *this;
  }

  template <class Mem>
  CUDA this_type& tell(AVar x, const sub_universe_type& dom, BInc<Mem>& has_changed) {
    sub->tell(x, dom, has_changed);
    return *this;
  }

  CUDA size_t to1D(int i, int j) const { return i *  num_columns() + j; }

private:
  template <class Alloc>
  CUDA local::BInc ask(const battery::vector<battery::vector<AVar, Alloc>, Alloc>& headers) const
  {
    for(int i = 0; i < headers.size(); ++i) {
      bool row_entailed = false;
      for(int j = 0; j < num_rows(); ++j) {
        row_entailed = true;
        for(int k = 0; k <  num_columns(); ++k) {
          if(!(sub->project(headers[i][k]) >= convert<IKind::ASK>(ask_table[to1D(j,k)]))) {
            row_entailed = false;
            break;
          }
        }
        if(row_entailed) {
          break;
        }
      }
      if(!row_entailed) { return false; }
    }
    return true;
  }

public:
  template <class Alloc>
  CUDA local::BInc ask(const ask_type<Alloc>& a) const {
    return ask(a.headers) && sub->ask(a.sub);
  }

private:
  template <class Mem>
  CUDA void refine(size_t table_num, size_t col, BInc<Mem>& has_changed) {
    sub_local_universe u{sub_local_universe::top()};
    for(int j = 0; j < num_rows(); ++j) {
      auto r = convert<IKind::TELL>(tell_table[to1D(j,col)]);
      if(!eliminated_rows[table_num].test(j)) {
        if(join(r, sub->project(headers[table_num][col])).is_top()) {
          eliminated_rows[table_num].set(j);
          has_changed.tell_top();
        }
        else {
          u.dtell(r);
        }
      }
    }
    sub->tell(headers[table_num][col], u, has_changed);
  }

public:
  CUDA size_t num_refinements() const {
    return
      sub->num_refinements() +
      headers.size() * num_columns();
  }

  template <class Mem>
  CUDA void refine(size_t i, BInc<Mem>& has_changed) {
    assert(i < num_refinements());
    if(i < sub->num_refinements()) {
      sub->refine(i, has_changed);
    }
    else {
      i -= sub->num_refinements();
      // refine(i /  num_columns(), i %  num_columns(), has_changed);
      refine(i % headers.size(), i / headers.size(), has_changed);
    }
  }

  template <class ExtractionStrategy = NonAtomicExtraction>
  CUDA bool is_extractable(const ExtractionStrategy& strategy = ExtractionStrategy()) const {
    // Check all remaining row are entailed.
    return ask(headers) && sub->is_extractable(strategy);
  }

  /** Extract an under-approximation if the last node popped \f$ a \f$ is an under-approximation.
   * If `B` is a search tree, the under-approximation consists in a search tree \f$ \{a\} \f$ with a single node, in that case, `ua` must be different from `top`. */
  template <class B>
  CUDA void extract(B& ua) const {
    if constexpr(impl::is_table_like<B>::value) {
      sub->extract(*ua.sub);
    }
    else {
      sub->extract(ua);
    }
  }

  CUDA sub_universe_type project(AVar x) const {
    return sub->project(x);
  }

  template<class Env>
  CUDA NI TFormula<typename Env::allocator_type> deinterpret(const Env& env) const {
    using F = TFormula<typename Env::allocator_type>;
    F sub_f = sub->deinterpret(env);
    typename F::Sequence seq{env.get_allocator()};
    if(sub_f.is(F::Seq) && sub_f.sig() == AND) {
      seq = std::move(sub_f.seq());
    }
    else {
      seq.push_back(std::move(sub_f));
    }
    for(int i = 0; i < headers.size(); ++i) {
      typename F::Sequence disjuncts{env.get_allocator()};
      for(int j = 0; j < num_rows(); ++j) {
        if(!eliminated_rows[i].test(j)) {
          typename F::Sequence conjuncts{env.get_allocator()};
          for(int k = 0; k < num_columns(); ++k) {
            if(!(sub->project(headers[i][k]) >= convert<IKind::ASK>(ask_table[to1D(j,k)]))) {
              conjuncts.push_back(tell_table[to1D(j,k)].deinterpret(headers[i][k], env));
            }
          }
          disjuncts.push_back(F::make_nary(AND, std::move(conjuncts), aty()));
        }
      }
      seq.push_back(F::make_nary(OR, std::move(disjuncts), aty()));
    }
    return F::make_nary(AND, std::move(seq));
  }
};

}

#endif
