// Copyright 2022 Pierre Talbot

#ifndef LALA_POWER_BAB_HPP
#define LALA_POWER_BAB_HPP

#include "battery/vector.hpp"
#include "battery/shared_ptr.hpp"
#include "lala/logic/logic.hpp"
#include "lala/universes/arith_bound.hpp"
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
  constexpr static const char* name = "BAB";

  template <class Alloc>
  struct tell_type {
    using sub_tell_type = sub_type::template tell_type<Alloc>;
    AVar x;
    bool optimization_mode;
    sub_tell_type sub_tell;
    tell_type(const Alloc& alloc = Alloc{}): sub_tell(alloc) {}
    tell_type(tell_type<Alloc>&&) = default;
    tell_type& operator=(tell_type<Alloc>&&) = default;
    tell_type(const tell_type<Alloc>&) = default;
    CUDA NI tell_type(AVar x, bool opt, const Alloc& alloc = Alloc{}):
      x(x), optimization_mode(opt), sub_tell(alloc) {}

    template <class BABTellType>
    CUDA NI tell_type(const BABTellType& other, const Alloc& alloc = Alloc{}):
      x(other.x), optimization_mode(other.optimization_mode),
      sub_tell(other.sub_tell, alloc)
    {}

    using allocator_type = Alloc;
    CUDA allocator_type get_allocator() const {
      return sub_tell.get_allocator();
    }

    template <class Alloc2>
    friend struct tell_type;
  };

  template <class Alloc2>
  using ask_type = typename sub_type::template ask_type<Alloc2>;

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
    AbstractDeps<Allocators...> deps_best(false, deps.template get_allocator<Allocators>()...);
    best = deps_best.template clone<best_type>(other.best);
  }

  CUDA AType aty() const {
    return atype;
  }

  CUDA allocator_type get_allocator() const {
    return sub->get_allocator();
  }

  CUDA local::B is_bot() const {
    return sub->is_bot();
  }

  CUDA local::B is_top() const {
    return x.is_untyped() && sub->is_top();
  }

public:
  template <bool diagnose = false, class F, class Env, class Alloc2>
  CUDA NI bool interpret_tell(const F& f, Env& env, tell_type<Alloc2>& tell, IDiagnostics& diagnostics) const {
    if(f.is_untyped() || f.type() == aty()) {
      if(f.is(F::Seq) && (f.sig() == MAXIMIZE || f.sig() == MINIMIZE)) {
        if(f.seq(0).is_variable()) {
          if(env.interpret(f.seq(0), tell.x, diagnostics)) {
            tell.optimization_mode = f.sig() == MINIMIZE;
            return true;
          }
          else {
            return false;
          }
        }
        // If the objective variable is already fixed to a constant, we ignore this predicate.
        // If there is only one objective, it becomes a satisfaction problem.
        else if(num_vars(f.seq(0)) == 0) {
          RETURN_INTERPRETATION_WARNING("This objective is already fixed to a constant, thus it is ignored.");
        }
        else {
          RETURN_INTERPRETATION_ERROR("Optimization predicates expect a variable to optimize (not an expression). Instead, you can create a new variable with the expression to optimize.");
        }
      }
      else if(f.type() == aty()) {
        RETURN_INTERPRETATION_ERROR("This formula has the type of BAB but it is not supported in this abstract domain.");
      }
    }
    return sub->template interpret_tell<diagnose>(f, env, tell.sub_tell, diagnostics);
  }

  template <bool diagnose = false, class F, class Env, class Alloc2>
  CUDA NI bool interpret_ask(const F& f, const Env& env, ask_type<Alloc2>& ask, IDiagnostics& diagnostics) const {
    return sub->template interpret_ask<diagnose>(f, env, ask, diagnostics);
  }

  template <IKind kind, bool diagnose = false, class F, class Env, class I>
  CUDA NI bool interpret(const F& f, Env& env, I& intermediate, IDiagnostics& diagnostics) const {
    if constexpr(kind == IKind::TELL) {
      return interpret_tell<diagnose>(f, env, intermediate, diagnostics);
    }
    else {
      return interpret_ask<diagnose>(f, env, intermediate, diagnostics);
    }
  }

  template <class Alloc>
  CUDA bool deduce(const tell_type<Alloc>& t) {
    bool has_changed = sub->deduce(t.sub_tell);
    if(!t.x.is_untyped()) {
      assert(x.is_untyped()); // multi-objective optimization not yet supported.
      x = t.x;
      optimization_mode = t.optimization_mode;
      return true;
    }
    return has_changed;
  }

  template <class Alloc2>
  CUDA NI TFormula<Alloc2> deinterpret_best_bound(const typename best_type::universe_type& best_bound, const Alloc2& alloc = Alloc2{}) const {
    using F = TFormula<Alloc2>;
    if((is_minimization() && best_bound.lb().is_top())
      ||(is_maximization() && best_bound.ub().is_top()))
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
  CUDA TFormula<Alloc2> deinterpret_best_bound(const Alloc2& alloc = Alloc2{}) const {
    return deinterpret_best_bound(best->project(x), alloc);
  }

  /** Update the variable to optimize `objective_var()` with a new bound. */
  CUDA local::B deduce(const typename best_type::universe_type& best_bound) {
    VarEnv<allocator_type> empty_env{};
    using F = TFormula<allocator_type>;
    F bound_formula = deinterpret_best_bound(best_bound, get_allocator());
    IDiagnostics diagnostics;
    typename sub_type::template tell_type<allocator_type> t;
    bool res = sub->interpret_tell(bound_formula, empty_env, t, diagnostics);
    assert(res);
    return sub->deduce(t);
  }

  /** Compare the best bound of two stores on the objective variable represented in this BAB abstract element.
   * \pre `is_optimization()` must be `true`.
   * \return `true` if `store1` is strictly better than `store2`, false otherwise.
  */
  template <class Store1, class Store2>
  CUDA bool compare_bound(const Store1& store1, const Store2& store2) const {
    assert(is_optimization());
    using Univ1 = typename Store1::local_universe;
    using Univ2 = typename Store2::local_universe;
    Univ1 bound1 = store1.project(x);
    Univ2 bound2 = store2.project(x);
    // When minimizing, the best bound is getting smaller and smaller.
    if(is_minimization()) {
      return (!bound1.is_top() && bound2.is_top()) || bound1.lb() > bound2.lb(); // note that `>` reflects the order of LB.
    }
    // And dually for maximization.
    else {
      return (!bound1.is_top() && bound2.is_top()) || bound1.ub() > bound2.ub();
    }
  }

  /** This deduction operator performs "branch-and-bound" by adding a constraint to the root node of the search tree to ensure the next solution is better than the current one, and store the best solution found.
   * \pre The current subelement must be extractable, and if it is an optimization problem, have a better bound than `best` (this is not checked here).
   * Beware this deduction operator is not idempotent (it must only be called once on each new solution).
   */
  CUDA local::B deduce() {
    sub->extract(*best);
    solutions_found++;
    if(is_optimization()) {
      deduce(best->project(x));
    }
    return true;
  }

  CUDA int solutions_count() const {
    return solutions_found;
  }

  /** Given an optimization problem, it is extractable only when we have explored the whole state space (indicated by the subdomain being equal to top), we have found one solution, and that solution is extractable. */
  template <class ExtractionStrategy = NonAtomicExtraction>
  CUDA bool is_extractable(const ExtractionStrategy& strategy = ExtractionStrategy()) const {
    return solutions_found > 0 && sub->is_bot() && best->is_extractable(strategy);
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
