# Neural Network from Scratch in Rust 🧠🦀

This project implements a simple feedforward neural network in pure Rust, with no external machine learning libraries. It learns logic gates such as the **CNOT**  using basic gradient descent and backpropagation.

---

## 🔧 Features

- Built entirely from scratch using [`ndarray`](https://docs.rs/ndarray/)
- Implements:
  - Dense layers with sigmoid activation
  - Binary cross-entropy loss
  - Gradient descent optimizer
- Learns 2-input/2-output (CNOT) logic
- Trains using mini-batch (1-sample) updates

---

## 📁 Project Structure

Neural-network-from-scratch/  
├── src/  
│ └── main.rs # Main implementation  
├── Cargo.toml # Dependencies and package config  
├── .gitignore # Ignores build artifacts

---
## 🚀 Running the Project

Make sure you have Rust and Cargo installed.

```bash
cargo run --release
```

	This will compile and run the neural network training loop. You’ll see the training loss and final predictions.

---
## 🧪 Example Logic Gate: **CNOT**

**CNOT** gate truth table:

|Input|Output|
|---|---|
|[1, 1, 0]|[1, 1, 1]|
|[1, 1, 1]|[1, 1, 0]|

The model will learn to mimic this logic through training.
