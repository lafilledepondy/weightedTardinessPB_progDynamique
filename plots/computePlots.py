import numpy as np
import matplotlib.pyplot as plt

from subgradients import (
	subgradient_basic,
	subgradient_Polyak,
	subgradient_ADS,
	cutting_planes,
)


def compare_step_sizes(step_sizes, problem_name,min_step_size=1e-6,
	initial_pi=None,
	initial_mu=None,
):
	if initial_pi is None:
		initial_pi = np.array([3.0, 1.0], dtype=float)
	if initial_mu is None:
		initial_mu = np.array([], dtype=float)

	results = {}

	for step in step_sizes:
		best_dual, _, history = subgradient_basic(
			initial_pi=np.copy(initial_pi),
			initial_mu=np.copy(initial_mu),
			min_step_size=min_step_size,
			initial_step_size=step,
			problem_name=problem_name,
		)

		dual_values = [it["dual_value"] for it in history]
		best_so_far = np.maximum.accumulate(np.array(dual_values, dtype=float)) if dual_values else np.array([])
		flatten_iteration = _find_flatten_iteration(best_so_far)
		results[step] = {
			"best_dual": best_dual,
			"dual_values": dual_values,
			"iterations": len(dual_values),
			"flatten_iteration": flatten_iteration,
		}

	return results


def compare_dual_value_methods(
	problem_name,
	min_step_size=1e-6,
	initial_pi=None,
	initial_mu=None,
	basic_kwargs=None,
	polyak_kwargs=None,
	ads_kwargs=None,
	cutting_planes_kwargs=None,
	plotTitle="Comparison of dual values",
	max_iterations=20,
	show_plot=True,
):
	"""Run several dual solvers and plot their dual value at each iteration.

	The plotted y-axis is the dual value and the x-axis is the iteration number.
	"""
	if initial_pi is None:
		initial_pi = np.array([3.0, 1.0], dtype=float)
	if initial_mu is None:
		initial_mu = np.array([], dtype=float)

	def _normalize_history(result):
		if result is None:
			return None

		if isinstance(result, tuple) and len(result) >= 3:
			best_dual, _, history = result[:3]
			dual_values = [it["dual_value"] for it in history]
			return {
				"best_dual": best_dual,
				"dual_values": dual_values,
				"iterations": len(dual_values),
			}

		if isinstance(result, dict):
			history = result.get("history", [])
			dual_values = result.get("dual_values")
			if dual_values is None:
				dual_values = [it["dual_value"] for it in history if "dual_value" in it]
			return {
				"best_dual": result.get("best_dual"),
				"dual_values": dual_values,
				"iterations": len(dual_values),
			}

		return None

	method_results = {}

	basic_kwargs = {} if basic_kwargs is None else dict(basic_kwargs)
	polyak_kwargs = {} if polyak_kwargs is None else dict(polyak_kwargs)
	ads_kwargs = {} if ads_kwargs is None else dict(ads_kwargs)
	cutting_planes_kwargs = {} if cutting_planes_kwargs is None else dict(cutting_planes_kwargs)
	if "epsilon" not in cutting_planes_kwargs:
		cutting_planes_kwargs["epsilon"] = 1e-6

	method_calls = {
		"basic": lambda: subgradient_basic(
			initial_pi=np.copy(initial_pi),
			initial_mu=np.copy(initial_mu),
			min_step_size=min_step_size,
			problem_name=problem_name,
			**basic_kwargs,
		),
		"polyak": lambda: subgradient_Polyak(
			initial_pi=np.copy(initial_pi),
			initial_mu=np.copy(initial_mu),
			min_step_size=min_step_size,
			problem_name=problem_name,
			**polyak_kwargs,
		),
		"ads": lambda: subgradient_ADS(
			initial_pi=np.copy(initial_pi),
			initial_mu=np.copy(initial_mu),
			min_step_size=min_step_size,
			problem_name=problem_name,
			**ads_kwargs,
		),
		"cutting_planes": lambda: cutting_planes(**cutting_planes_kwargs),
	}

	for method_name, solver_call in method_calls.items():
		try:
			method_results[method_name] = _normalize_history(solver_call())
		except (NotImplementedError, TypeError, ValueError):
			method_results[method_name] = None

	plot_dual_value_comparison(
		method_results,
		plotTitle=plotTitle,
		max_iterations=max_iterations,
		show_plot=show_plot,
	)
	return method_results


def _find_flatten_iteration(best_so_far, window=5, tol=1e-4):
	"""Return the first iteration where improvement over a recent window is tiny.
	If the curve never flattens, return None.
	"""
	if best_so_far is None or len(best_so_far) < window + 1:
		return None

	for idx in range(window, len(best_so_far)):
		improvement = best_so_far[idx] - best_so_far[idx - window]
		if abs(improvement) <= tol:
			return idx + 1  # 1-based iteration number

	return None


def plot_dual_value_comparison(
	results,
	plotTitle="Comparison of dual values",
	max_iterations=20,
	show_plot=True,
	mode="small_multiples",
):
	valid_results = [(name, data) for name, data in results.items() if data and data.get("dual_values")]
	if not valid_results:
		return None

	if mode == "overlay":
		plt.figure(figsize=(10, 6))
		for method_name, data in valid_results:
			y = np.array(data["dual_values"][:max_iterations], dtype=float)
			x = np.arange(1, len(y) + 1)
			label = method_name
			best_dual = data.get("best_dual")
			if best_dual is not None:
				label += f" | best={best_dual}"
			plt.plot(x, y, label=label, linewidth=2, alpha=0.85)

		plt.title(f"{plotTitle} (first {max_iterations} iterations)")
		plt.xlabel("Iteration")
		plt.ylabel("Dual value")
		plt.grid(True, linestyle="--", alpha=0.4)
		plt.legend()
		plt.tight_layout()
		if show_plot:
			plt.show()
		return plt.gca()

	# default: one panel per method to avoid line overlap
	n_methods = len(valid_results)
	ncols = 2 if n_methods > 1 else 1
	nrows = int(np.ceil(n_methods / ncols))
	fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 4 * nrows), squeeze=False)
	axes_flat = axes.flatten()

	for idx, (method_name, data) in enumerate(valid_results):
		ax = axes_flat[idx]
		y = np.array(data["dual_values"][:max_iterations], dtype=float)
		x = np.arange(1, len(y) + 1)
		best_so_far = np.maximum.accumulate(y)

		ax.plot(x, y, color="#1f77b4", linewidth=1.8, alpha=0.9, label="dual")
		ax.plot(x, best_so_far, color="#ff7f0e", linewidth=1.8, linestyle="--", label="best-so-far")

		if len(y) > 0:
			ax.scatter([x[-1]], [y[-1]], color="#1f77b4", s=25)
			ax.scatter([x[-1]], [best_so_far[-1]], color="#ff7f0e", s=25)

		best_dual = data.get("best_dual")
		title = method_name if best_dual is None else f"{method_name} | best={best_dual}"
		ax.set_title(title)
		ax.set_xlabel("Iteration")
		ax.set_ylabel("Dual value")
		ax.grid(True, linestyle="--", alpha=0.35)
		ax.legend(loc="best")

	for idx in range(n_methods, len(axes_flat)):
		axes_flat[idx].axis("off")

	fig.suptitle(f"{plotTitle} (first {max_iterations} iterations)", y=0.995)
	fig.tight_layout()
	if show_plot:
		plt.show()
	return axes


def plot_step_size_comparison(results, plotTitle="Comparison de step sizes", max_iterations=20):
	plt.figure(figsize=(10, 6))

	for step, data in results.items():
		y = data["dual_values"][:max_iterations]
		x = np.arange(1, len(y) + 1)
		flatten_iteration = data.get("flatten_iteration")
		label = f"step={step} | best={data['best_dual']}"
		if flatten_iteration is not None:
			label += f" | flat@{flatten_iteration}"
		plt.plot(x, y, label=label)
		if flatten_iteration is not None and flatten_iteration <= max_iterations:
			flat_idx = flatten_iteration - 1
			if flat_idx < len(y):
				plt.scatter([flatten_iteration], [y[flat_idx]], s=30)

	plt.title(f"{plotTitle} (first {max_iterations} iterations)")
	plt.xlabel("Iteration")
	plt.ylabel("Dual value")
	plt.grid(True, linestyle="--", alpha=0.4)
	plt.legend()
	plt.tight_layout()
	plt.show()


def main():
	compare_dual_value_methods(
		problem_name="P2",
		min_step_size=1e-6,
		plotTitle="Dual value comparison across methods",
		max_iterations=20,
	)

if __name__ == "__main__":
	main()
