# 🧭 Roadmap: ProtoSynth Development

This roadmap outlines the phases of building a self-modifying, compression-driven AI prototype using a Lisp-like interpreter in Python ASTs.

---

## ✅ Phase 1: Core Interpreter & Self-Inspection (Week 1)

- [x] Define minimal Lisp-style AST node class (`LispNode`)
- [x] Implement resource-constrained interpreter (`LispInterpreter`)
- [x] Add primitive operations (+, -, *, /, ==, <, if, let, var)
- [x] Support for nested expressions and variable binding
- [x] Implement `get_self_ast()` for self-inspection
- [ ] Add serialization (printable/exportable representation of AST)

---

## 🚧 Phase 2: Mutation Engine & Syntactic Verification (Week 2)

- [ ] Walk and traverse AST structure safely
- [ ] Define basic mutation operations:
  - Replace operator
  - Swap constants
  - Change variable names
  - Inject random subtrees
- [ ] Add syntactic validity checks:
  - Balanced expression trees
  - Variable binding consistency
  - Operation arity enforcement

---

## 🧪 Phase 3: Evaluation & Fitness (Week 3)

- [ ] Create symbolic pattern environments (binary streams, logic gates)
- [ ] Define fitness scoring metric:
  - Compression ratio
  - Prediction accuracy
- [ ] Track success rate across generations
- [ ] Create generation loop (mutate → verify → evaluate → select)

---

## 📚 Phase 4: Emergent Modularity & Memory (Week 4+)

- [ ] Store high-utility subtrees as reusable modules
- [ ] Allow recombination of verified submodules
- [ ] Track lineage/evolution of modules
- [ ] Explore curriculum generation or world transitions

---

## 🧠 Stretch Goals & Future Experiments

- [ ] Self-hosting: system mutates its own interpreter logic
- [ ] Type system & dependent type verification
- [ ] Multi-agent populations with variation strategies
- [ ] Transition to 2D physics or cellular automata environments
- [ ] Web-based visualizer for AST evolution

---

