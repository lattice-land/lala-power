// Copyright 2023 Pierre Talbot

#ifndef LALA_POWER_SPLIT_STRATEGY_HPP
#define LALA_POWER_SPLIT_STRATEGY_HPP

#include "battery/vector.hpp"
#include "battery/shared_ptr.hpp"
#include "branch.hpp"
#include "lala/logic/logic.hpp"
#include "lala/abstract_deps.hpp"

namespace lala {

enum class VariableOrder {
  INPUT_ORDER,
  FIRST_FAIL,
  ANTI_FIRST_FAIL,
  SMALLEST,
  LARGEST,
  // unsupported:
  // OCCURRENCE,
  // MOST_CONSTRAINED,
  // MAX_REGRET,
  // DOM_W_DEG,
  // RANDOM
};

enum class ValueOrder {
  MIN,
  MAX,
  MEDIAN,
  SPLIT,
  REVERSE_SPLIT,
  // unsupported:
  // INTERVAL,
  // RANDOM,
  // MIDDLE,
};

template <class A, class Allocator = typename A::allocator_type>
class SplitStrategy {
public:
  using allocator_type = Allocator;
  using sub_type = A;
  using sub_allocator_type = typename sub_type::allocator_type;
  using sub_tell_type = sub_type::template tell_type<allocator_type>;
  using branch_type = Branch<sub_tell_type, allocator_type>;
  using this_type = SplitStrategy<sub_type, allocator_type>;

  constexpr static const bool is_abstract_universe = false;
  constexpr static const bool sequential = sub_type::sequential;
  constexpr static const bool is_totally_ordered = false;
  constexpr static const bool preserve_bot = true;
  constexpr static const bool preserve_top = true;
  // The next properties should be checked more seriously, relying on the sub-domain might be uneccessarily restrictive.
  constexpr static const bool preserve_join = sub_type::preserve_join;
  constexpr static const bool preserve_meet = sub_type::preserve_meet;
  constexpr static const bool injective_concretization = sub_type::injective_concretization;
  constexpr static const bool preserve_concrete_covers = sub_type::preserve_concrete_covers;

  constexpr static const char* name = "SplitStrategy";

  template <class Alloc>
  struct snapshot_type {
    size_t num_strategies;
    int current_strategy;
    int next_unassigned_var;

    CUDA snapshot_type(size_t num_strategies, int current_strategy, int next_unassigned_var)
      : num_strategies(num_strategies)
      , current_strategy(current_strategy)
      , next_unassigned_var(next_unassigned_var)
    {}

    snapshot_type(const snapshot_type<Alloc>&) = default;
    snapshot_type(snapshot_type<Alloc>&&) = default;
    snapshot_type<Alloc>& operator=(snapshot_type<Alloc>&&) = default;
    snapshot_type<Alloc>& operator=(const snapshot_type<Alloc>&) = default;

    template<class SplitSnapshot>
    CUDA snapshot_type(const SplitSnapshot& other, const Alloc&)
      : num_strategies(other.num_strategies)
      , current_strategy(other.current_strategy)
      , next_unassigned_var(other.next_unassigned_var)
    {}
  };

  /** A split strategy consists of a variable order and value order on a subset of the variables. */
  template <class Alloc2>
  struct strategy_type {
    VariableOrder var_order;
    ValueOrder val_order;
    battery::vector<AVar, Alloc2> vars;

    CUDA strategy_type(const Alloc2& alloc = Alloc2{}): var_order(VariableOrder::INPUT_ORDER), val_order(ValueOrder::MIN), vars(alloc) {}
    strategy_type(const strategy_type<Alloc2>&) = default;
    strategy_type(strategy_type<Alloc2>&&) = default;
    strategy_type& operator=(strategy_type<Alloc2>&&) = default;

    CUDA strategy_type(VariableOrder var_order, ValueOrder val_order, battery::vector<AVar, Alloc2>&& vars)
      : var_order(var_order), val_order(val_order), vars(std::move(vars)) {}

    using allocator_type = Alloc2;
    CUDA allocator_type get_allocator() const {
      return vars.get_allocator();
    }

    template <class StrategyType>
    CUDA strategy_type(const StrategyType& other, const Alloc2& alloc = Alloc2{})
    : var_order(other.var_order), val_order(other.val_order), vars(other.vars, alloc) {}

    template <class Alloc3>
    friend class strategy_type;
  };

  template <class Alloc2>
  using tell_type = battery::vector<strategy_type<Alloc2>, Alloc2>;

  template <class A2, class Alloc2>
  friend class SplitStrategy;

private:
  using universe_type = typename A::universe_type;
  using LB = typename universe_type::LB;
  using UB = typename universe_type::UB;

  AType atype;
  abstract_ptr<A> a;
  battery::vector<strategy_type<allocator_type>, allocator_type> strategies;
  int current_strategy;
  int next_unassigned_var;

  CUDA const battery::vector<AVar, allocator_type>& current_vars() const {
    return strategies[current_strategy].vars;
  }

  CUDA NI void move_to_next_unassigned_var() {
    while(current_strategy < strategies.size()) {
      const auto& vars = current_vars();
      while(next_unassigned_var < vars.size()) {
        universe_type v{};
        a->project(vars[next_unassigned_var], v);
        if(dual<UB>(v.lb()) < v.ub()) {
          return;
        }
        next_unassigned_var++;
      }
      current_strategy++;
      next_unassigned_var = 0;
    }
  }

  template <class MapFunction>
  CUDA NI AVar var_map_fold_left(const battery::vector<AVar, allocator_type>& vars, MapFunction op) {
    int i = next_unassigned_var;
    int best_i = i;
    auto best = op(a->project(vars[i]));
    for(++i; i < vars.size(); ++i) {
      const auto& u = a->project(vars[i]);
      if(dual<UB>(u.lb()) < u.ub()) {
        if(best.meet(op(u))) {
          best_i = i;
        }
      }
    }
    return vars[best_i];
  }

  CUDA AVar select_var() {
    const auto& strat = strategies[current_strategy];
    const auto& vars = strat.vars;
    switch(strat.var_order) {
      case VariableOrder::INPUT_ORDER: return vars[next_unassigned_var];
      case VariableOrder::FIRST_FAIL: return var_map_fold_left(vars, [](const universe_type& u) { return u.width().ub(); });
      case VariableOrder::ANTI_FIRST_FAIL: return var_map_fold_left(vars, [](const universe_type& u) { return dual<LB>(u.width().ub()); });
      case VariableOrder::LARGEST: return var_map_fold_left(vars, [](const universe_type& u) { return dual<LB>(u.ub()); });
      case VariableOrder::SMALLEST: return var_map_fold_left(vars, [](const universe_type& u) { return dual<UB>(u.lb()); });
      default: printf("BUG: unsupported variable order strategy\n"); assert(false); return AVar{};
    }
  }

  template <class U>
  CUDA NI branch_type make_branch(AVar x, Sig left_op, Sig right_op, const U& u) {
    if((u.is_top() && U::preserve_top) || (u.is_bot() && U::preserve_bot)) {
      if(u.is_top()) {
        printf("%% WARNING: Cannot currently branch on unbounded variables.\n");
      }
      return branch_type(get_allocator());
    }
    using F = TFormula<allocator_type>;
    using branch_vector = battery::vector<sub_tell_type, allocator_type>;
    VarEnv<allocator_type> empty_env{};
    auto k = u.template deinterpret<F>();
    IDiagnostics diagnostics;
    sub_tell_type left(get_allocator());
    sub_tell_type right(get_allocator());
    bool res = a->interpret_tell(F::make_binary(F::make_avar(x), left_op, k, x.aty(), get_allocator()), empty_env, left, diagnostics);
    res &= a->interpret_tell(F::make_binary(F::make_avar(x), right_op, k, x.aty(), get_allocator()), empty_env, right, diagnostics);
    if(res) {
      return Branch(branch_vector({std::move(left), std::move(right)}, get_allocator()));
    }
    // Fallback on a more standard split search strategy.
    // We don't print anything because it might interfere with the output (without lock).
    else if(left_op != LEQ || right_op != GT) {
      return make_branch(x, LEQ, GT, u);
    }
    else {
      printf("%% WARNING: The subdomain does not support the underlying search strategy.\n");
      // Diagnostics mode.
      a->template interpret_tell<true>(F::make_binary(F::make_avar(x), left_op, k, x.aty(), get_allocator()), empty_env, left, diagnostics);
      a->template interpret_tell<true>(F::make_binary(F::make_avar(x), right_op, k, x.aty(), get_allocator()), empty_env, right, diagnostics);
      diagnostics.print();
      return branch_type(get_allocator());
    }
  }

public:
  CUDA SplitStrategy(AType atype, abstract_ptr<A> a, const allocator_type& alloc = allocator_type()):
    atype(atype), a(a), current_strategy(0), next_unassigned_var(0), strategies(alloc)
  {}

  template<class A2, class Alloc2, class... Allocators>
  CUDA SplitStrategy(const SplitStrategy<A2, Alloc2>& other, AbstractDeps<Allocators...>& deps)
   : atype(other.atype),
     a(deps.template clone<A>(other.a)),
     strategies(other.strategies, deps.template get_allocator<allocator_type>()),
     current_strategy(other.current_strategy),
     next_unassigned_var(other.next_unassigned_var)
  {}

  CUDA AType aty() const {
    return atype;
  }

  CUDA allocator_type get_allocator() const {
    return strategies.get_allocator();
  }

  template <class Alloc2 = allocator_type>
  CUDA snapshot_type<Alloc2> snapshot(const Alloc2& alloc = Alloc2()) const {
    return snapshot_type<Alloc2>{strategies.size(), current_strategy, next_unassigned_var};
  }

  template <class Alloc2 = allocator_type>
  CUDA void restore(const snapshot_type<Alloc2>& snap) {
    while(strategies.size() > snap.num_strategies) {
      strategies.pop_back();
    }
    current_strategy = snap.current_strategy;
    next_unassigned_var = snap.next_unassigned_var;
  }

  /** Restart the search from the first variable. */
  CUDA void reset() {
    current_strategy = 0;
    next_unassigned_var = 0;
  }

  /** This interpretation function expects `f` to be a predicate of the form `search(VariableOrder, ValueOrder, x_1, x_2, ..., x_n)`. */
  template <bool diagnose = false, class F, class Env, class Alloc2>
  CUDA NI bool interpret_tell(const F& f, Env& env, tell_type<Alloc2>& tell, IDiagnostics& diagnostics) const {
    if(!(f.is(F::ESeq)
      && f.eseq().size() >= 2
      && f.esig() == "search"
      && f.eseq()[0].is(F::ESeq) && f.eseq()[0].eseq().size() == 0
      && f.eseq()[1].is(F::ESeq) && f.eseq()[1].eseq().size() == 0))
    {
      RETURN_INTERPRETATION_ERROR("SplitStrategy can only interpret predicates of the form `search(input_order, indomain_min, x1, ..., xN)`.");
    }
    strategy_type<Alloc2> strat{tell.get_allocator()};
    const auto& var_order_str = f.eseq()[0].esig();
    const auto& val_order_str = f.eseq()[1].esig();
    if(var_order_str == "input_order") { strat.var_order = VariableOrder::INPUT_ORDER; }
    else if(var_order_str == "first_fail") { strat.var_order = VariableOrder::FIRST_FAIL; }
    else if(var_order_str == "anti_first_fail") { strat.var_order = VariableOrder::ANTI_FIRST_FAIL; }
    else if(var_order_str == "smallest") { strat.var_order = VariableOrder::SMALLEST; }
    else if(var_order_str == "largest") { strat.var_order = VariableOrder::LARGEST; }
    else {
      RETURN_INTERPRETATION_ERROR("This variable order strategy is unsupported.");
    }
    if(val_order_str == "indomain_min") { strat.val_order = ValueOrder::MIN; }
    else if(val_order_str == "indomain_max") { strat.val_order = ValueOrder::MAX; }
    else if(val_order_str == "indomain_median") { strat.val_order = ValueOrder::MEDIAN; }
    else if(val_order_str == "indomain_split") { strat.val_order = ValueOrder::SPLIT; }
    else if(val_order_str == "indomain_reverse_split") { strat.val_order = ValueOrder::REVERSE_SPLIT; }
    else {
      RETURN_INTERPRETATION_ERROR("This value order strategy is unsupported.");
    }
    for(int i = 2; i < f.eseq().size(); ++i) {
      if(f.eseq(i).is(F::LV)) {
        strat.vars.push_back(AVar{});
        if(!env.interpret(f.eseq(i), strat.vars.back(), diagnostics)) {
          return false;
        }
      }
      else if(f.eseq(i).is(F::V)) {
        strat.vars.push_back(f.eseq(i).v());
      }
      else if(num_vars(f.eseq(i)) > 0) {
        RETURN_INTERPRETATION_ERROR("The predicate `search` only supports variables or constants, but an expression containing a variable was passed to it.");
      }
      // Ignore constant expressions.
      else {}
    }
    if(strat.vars.size() == 0) {
      RETURN_INTERPRETATION_WARNING("The predicate `search` has no variable, and thus it is ignored.");
    }
    tell.push_back(std::move(strat));
    return true;
  }

  template <IKind kind, bool diagnose = false, class F, class Env, class I>
  CUDA bool interpret(const F& f, Env& env, I& intermediate, IDiagnostics& diagnostics) const {
    static_assert(kind == IKind::TELL);
    return interpret_tell<diagnose>(f, env, intermediate, diagnostics);
  }

  /** This deduce method adds new strategies, and therefore do not satisfy the PCCP model.
   * Calling this function multiple times will create multiple strategies, that will be called in sequence along a branch of the search tree.
   * @sequential
  */
  template <class Alloc2>
  CUDA bool deduce(const tell_type<Alloc2>& t) {
    bool has_changed = false;
    for(int i = 0; i < t.size(); ++i) {
      if(t[i].vars.size() > 0) {
        strategies.push_back(t[i]);
        has_changed = true;
      }
    }
    return has_changed;
  }

  /** Split the next unassigned variable according to the current strategy.
   * If all variables of the current strategy are assigned, use the next strategy.
   * If no strategy remains, returns an empty set of branches.

   If the next unassigned variable cannot be split, for instance because the value ordering strategy maps to `bot` or `top`, an empty set of branches is returned.
   This also means that you cannot suppose `split(a) = {}` to mean `a` is at `bot`. */
  CUDA NI branch_type split() {
    if(a->is_bot()) {
      return branch_type(get_allocator());
    }
    move_to_next_unassigned_var();
    if(current_strategy < strategies.size()) {
      AVar x = select_var();
      switch(strategies[current_strategy].val_order) {
        case ValueOrder::MIN: return make_branch(x, EQ, GT, a->project(x).lb());
        case ValueOrder::MAX: return make_branch(x, EQ, LT, a->project(x).ub());
        case ValueOrder::MEDIAN: return make_branch(x, EQ, NEQ, a->project(x).median().lb());
        case ValueOrder::SPLIT: return make_branch(x, LEQ, GT, a->project(x).median().lb());
        case ValueOrder::REVERSE_SPLIT: return make_branch(x, GT, LEQ, a->project(x).median().lb());
        default: printf("BUG: unsupported value order strategy\n"); assert(false); return branch_type(get_allocator());
      }
    }
    else {
      return branch_type(get_allocator());
    }
  }

  CUDA size_t num_strategies() const {
    return strategies.size();
  }
};

}

#endif