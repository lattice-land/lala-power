// Copyright 2022 Pierre Talbot

#ifndef LALA_POWER_BAB_HPP
#define LALA_POWER_BAB_HPP

#include "battery/vector.hpp"
#include "battery/shared_ptr.hpp"
#include "lala/logic/logic.hpp"
#include "lala/universes/primitive_upset.hpp"
#include "lala/abstract_deps.hpp"
#include "lala/vstore.hpp"

namespace lala {
template <class A, class B> class BAB;
namespace impl {
  template <class>
  struct is_bab_like {
    static constexpr bool value = false;
  };
  template<class A, class B>
  struct is_bab_like<BAB<A, B>> {
    static constexpr bool value = true;
  };
}

template <class A, class B = A>
class BAB {
public:
  using allocator_type = typename A::allocator_type;
  using sub_type = A;
  using sub_ptr = abstract_ptr<sub_type>;
  using best_type = B;
  using best_ptr = abstract_ptr<best_type>;
  using this_type = BAB<sub_type, best_type>;

  template <class Alloc2>
  struct tell_type {
    using sub_tell_type = sub_type::template tell_type<Alloc2>;
    AVar x;
    bool optimization_mode;
    battery::vector<sub_tell_type, Alloc2> sub_tells;
    tell_type() = default;
    tell_type(tell_type<Alloc2>&&) = default;
    tell_type(const tell_type<Alloc2>&) = default;
    CUDA NI tell_type(AVar x, bool opt, const Alloc2& alloc = Alloc2()):
      x(x), optimization_mode(opt), sub_tells(alloc) {}

    template <class BABTellType>
    CUDA NI tell_type(const BABTellType& other, const Alloc2& alloc = Alloc2()):
      x(other.x), optimization_mode(other.optimization_mode),
      sub_tells(other.sub_tells, alloc)
    {}

    template <class Alloc3>
    friend struct tell_type;
  };

  template <class Alloc2>
  using ask_type = battery::vector<typename sub_type::template ask_type<Alloc2>, Alloc2>;

  template<class F, class Env>
  using iresult_tell = IResult<tell_type<typename Env::allocator_type>, F>;

  template<class F, class Env>
  using iresult_ask = IResult<ask_type<typename Env::allocator_type>, F>;

  constexpr static const char* name = "BAB";

  template <class A2, class B2>
  friend class BAB;

private:
  AType atype;
  sub_ptr sub;
  best_ptr best;
  AVar x;
  bool optimization_mode; // `true` for minimization, `false` for maximization.
  int solutions_found;

public:
  CUDA BAB(AType atype, sub_ptr sub, best_ptr best)
   : atype(atype), sub(std::move(sub)), best(std::move(best)), x(),
     solutions_found(0)
  {
    assert(this->sub);
    assert(this->best);
  }

  /** Construct BAB by copying `other`.
   * The best solution is copied using a fresh AbstractDeps, and thus is not intended to be shared with other abstract domains.
   * For instance, if `best` is a VStore, it shares the same AType than the VStore underlying `sub`.
   * Hence, if we copy it using `deps`, both VStore will be shared which is not the intended behavior.
   */
  template<class A2, class B2, class... Allocators>
  CUDA NI BAB(const BAB<A2, B2>& other, AbstractDeps<Allocators...>& deps)
   : atype(other.atype)
   , sub(deps.template clone<sub_type>(other.sub))
   , x(other.x)
   , optimization_mode(other.optimization_mode)
  {
    AbstractDeps<Allocators...> deps_best(deps.template get_allocator<Allocators>()...);
    best = deps_best.template clone<best_type>(other.best);
  }

  CUDA AType aty() const {
    return atype;
  }

  CUDA allocator_type get_allocator() const {
    return sub->get_allocator();
  }

  CUDA local::BInc is_top() const {
    return sub->is_top();
  }

  CUDA local::BDec is_bot() const {
    return x.is_untyped() && sub->is_bot();
  }

private:
  template <bool is_tell, class R, class SubR, class F, class Env>
  CUDA void interpret_sub(R& res, SubR& sub_res, const F& f, Env& env) {
    if(!res.has_value()) {
      return;
    }
    if(sub_res.has_value()) {
      if constexpr(is_tell) {
        res.value().sub_tells.push_back(std::move(sub_res.value()));
      }
      else {
        res.value().push_back(std::move(sub_res.value()));
      }
      res.join_warnings(std::move(sub_res));
    }
    else {
      res.join_errors(std::move(sub_res));
    }
  }

  template <bool is_tell, class R, class F, class Env>
  CUDA void interpret_sub(R& res, const F& f, Env& env) {
    if constexpr(is_tell){
      auto sub_res = sub->interpret_tell_in(f, env);
      interpret_sub<is_tell>(res, sub_res, f, env);
    }
    else {
      auto sub_res = sub->interpret_ask_in(f, env);
      interpret_sub<is_tell>(res, sub_res, f, env);
    }
  }

  template <class F, class Env>
  CUDA NI void interpret_optimization_predicate(iresult_tell<F, Env>& res, const F& f, Env& env) {
    if(f.is_untyped() || f.type() == aty()) {
      if(f.is(F::Seq) && (f.sig() == MAXIMIZE || f.sig() == MINIMIZE)) {
        if(f.seq(0).is_variable()) {
          res.value().optimization_mode = f.sig() == MINIMIZE;
          auto var_res = env.interpret(f.seq(0));
          if(var_res.has_value()) {
            res.value().x = var_res.value();
          }
          else {
            res.join_errors(std::move(var_res));
          }
        }
        // If the objective variable is already fixed to a constant, we ignore this predicate.
        // If there is only one objective, it becomes a satisfaction problem.
        else if(num_vars(f.seq(0)) == 0) {
          return;
        }
        else {
          res = iresult_tell<F, Env>(IError<F>(true, name, "Optimization predicates expect a variable to optimize. Instead, you can create a new variable with the expression to optimize.", f));
        }
        return;
      }
      else if(f.type() == aty()) {
        res = iresult_tell<F, Env>(IError<F>(true, name, "Unsupported formula.", f));
        return;
      }
    }
    interpret_sub<true>(res, f, env);
  }

  template <class R, class F, class Env>
  CUDA NI void interpret_tell_in(R& res, const F& f, Env& env) {
    if(!res.has_value()) {
      return;
    }
    if(f.is_untyped() || f.type() == aty()) {
      if(f.is(F::Seq) && f.sig() == AND) {
        for(int i = 0; i < f.seq().size() && res.has_value(); ++i) {
          interpret_tell_in(res, f.seq(i), env);
        }
      }
      else {
        interpret_optimization_predicate(res, f, env);
      }
    }
    else {
      interpret_sub<true>(res, f, env);
    }
  }

public:
  template <class F, class Env>
  CUDA NI iresult_tell<F, Env> interpret_tell_in(const F& f, Env& env) {
    iresult_tell<F, Env> res(tell_type<typename Env::allocator_type>{});
    interpret_tell_in(res, f, env);
    return std::move(res);
  }

  template <class F, class Env>
  CUDA NI iresult_ask<F, Env> interpret_ask_in(const F& f, Env& env) {
    iresult_ask<F, Env> res{env.get_allocator()};
    interpret_sub<false>(res, f, env);
    return std::move(res);
  }

  template <class Alloc, class Mem>
  CUDA this_type& tell(const tell_type<Alloc>& t, BInc<Mem>& has_changed) {
    for(int i = 0; i < t.sub_tells.size(); ++i) {
      sub->tell(t.sub_tells[i], has_changed);
    }
    if(!t.x.is_untyped()) {
      assert(x.is_untyped()); // multi-objective optimization not yet supported.
      x = t.x;
      optimization_mode = t.optimization_mode;
      has_changed.tell_top();
    }
    return *this;
  }

  template <class Alloc2>
  CUDA NI TFormula<Alloc2> deinterpret_best_bound(const typename best_type::universe_type& best_bound, const Alloc2& alloc = Alloc2()) const {
    using F = TFormula<Alloc2>;
    if((is_minimization() && best_bound.lb().is_bot())
      ||(is_maximization() && best_bound.ub().is_bot()))
    {
      return F::make_true();
    }
    Sig optimize_sig = is_minimization() ? LT : GT;
    F constant = is_minimization()
      ? best_bound.lb().template deinterpret<F>()
      : best_bound.ub().template deinterpret<F>();
    return F::make_binary(F::make_avar(x), optimize_sig, constant, UNTYPED, alloc);
  }

  template <class Alloc2>
  CUDA TFormula<Alloc2> deinterpret_best_bound(const Alloc2& alloc = Alloc2()) const {
    return deinterpret_best_bound(best->project(x), alloc);
  }

  /** Update the variable to optimize `objective_var()` with a new bound. */
  template <class Mem>
  CUDA this_type& tell(const typename best_type::universe_type& best_bound, BInc<Mem>& has_changed) {
    VarEnv<allocator_type> empty_env{};
    auto bound_formula = deinterpret_best_bound(best_bound, get_allocator());
    auto t = sub->interpret_tell_in(bound_formula, empty_env).value();
    sub->tell(t, has_changed);
    return *this;
  }

  /** Compare the best bound of two stores on the objective variable represented in this BAB abstract element.
   * \pre `is_optimization()` must be `true`.
   * \return `true` if `store1` is strictly better than `store2`, false otherwise.
  */
  template <class Store1, class Store2>
  CUDA bool compare_bound(const Store1& store1, const Store2& store2) const {
    assert(is_optimization());
    const auto& bound1 = store1.project(x);
    const auto& bound2 = store2.project(x);
    using LB = typename Store1::universe_type::LB;
    using UB = typename Store1::universe_type::UB;
    // When minimizing, the best bound is getting smaller and smaller, hence the order over LB is not suited, we must compare the bound in UB which represents this fact.
    // And dually for maximization.
    if(is_minimization()) {
      return dual<UB>(bound1.lb()) > dual<UB>(bound2.lb());
    }
    else {
      return dual<LB>(bound1.ub()) > dual<LB>(bound2.ub());
    }
  }

  /** This refinement operator performs "branch-and-bound" by adding a constraint to the root node of the search tree to ensure the next solution is better than the current one, and store the best solution found.
   * \pre The current subelement must be extractable, and if it is an optimization problem, have a better bound than `best` (this is not checked here).
   * Beware this refinement operator is not idempotent (it must only be called once on each new solution).
   */
  template <class Mem>
  CUDA void refine(BInc<Mem>& has_changed) {
    sub->extract(*best);
    solutions_found++;
    if(is_optimization()) {
      tell(best->project(x), has_changed);
    }
  }

  CUDA int solutions_count() const {
    return solutions_found;
  }

  /** Given an optimization problem, it is extractable only when we have explored the whole state space (indicated by the subdomain being equal to top), we have found one solution, and that solution is extractable. */
  template <class ExtractionStrategy = NonAtomicExtraction>
  CUDA bool is_extractable(const ExtractionStrategy& strategy = ExtractionStrategy()) const {
    return solutions_found > 0 && sub->is_top() && best->is_extractable(strategy);
  }

  /** Extract the best solution found in `ua`.
   * \pre `is_extractable()` must return `true`. */
  template <class AbstractBest>
  CUDA void extract(AbstractBest& ua) const {
    if constexpr(impl::is_bab_like<AbstractBest>::value) {
      best->extract(*(ua.best));
      ua.solutions_found = solutions_found;
      ua.x = x;
      ua.optimization_mode = optimization_mode;
    }
    else {
      return best->extract(ua);
    }
  }

  /** If `is_extractable()` is not `true`, the returned element might not be an optimum, and should be seen as the best optimum found so far.
  */
  CUDA const best_type& optimum() const {
    return *best;
  }

  CUDA best_ptr optimum_ptr() const {
    return best;
  }

  CUDA bool is_satisfaction() const {
    return x.is_untyped();
  }

  CUDA bool is_optimization() const {
    return !is_satisfaction();
  }

  CUDA bool is_minimization() const {
    return is_optimization() && optimization_mode;
  }

  CUDA bool is_maximization() const {
    return is_optimization() && !optimization_mode;
  }

  CUDA AVar objective_var() const {
    return x;
  }
};

}

#endif
