.PHONY: test smoke results release

test:
	pytest -q tests && python scripts/staged_parity_harness.py

smoke:
	protosynth-evolve --env periodic_k4 --gens 40 --k 4 --mu 8 --lambda 16 --seed 1
	protosynth-evolve --env markov_k2   --gens 40 --k 2 --mu 8 --lambda 16 --seed 2

results:
	python scripts/plot_results.py --out results/v0.1.0/

release: test smoke
	git tag v0.1.0 && git push --tags
