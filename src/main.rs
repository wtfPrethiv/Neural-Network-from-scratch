use ndarray::{Array1, Array2, Axis};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;


fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}

fn sigmoid_derivative(z: f64) -> f64 {
    let s = sigmoid(z);
    s * (1.0 - s)
}

fn dense(a_in: &Array1<f64>, w: &Array2<f64>, b: &Array1<f64>) -> (Array1<f64>, Array1<f64>) {
    let z = w.t().dot(a_in) + b;
    let a = z.mapv(sigmoid);
    (z, a)
}

fn model(
    x: &Array1<f64>,
    w1: &Array2<f64>, b1: &Array1<f64>,
    w2: &Array2<f64>, b2: &Array1<f64>,
) -> (Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>) {
    let (z1, a1) = dense(x, w1, b1);
    let (z2, a2) = dense(&a1, w2, b2);
    (z1, a1, z2, a2)
}

fn binary_cross_entropy(y_hat: &Array1<f64>, y: &Array1<f64>) -> f64 {
    let eps = 1e-15;
    let loss = y.iter()
        .zip(y_hat.iter())
        .map(|(&y, &y_hat)| {
            y * (y_hat + eps).ln() + (1.0 - y) * (1.0 - y_hat + eps).ln()
        })
        .sum::<f64>();
    -loss
}

fn train_step(
    x: &Array1<f64>, y: &Array1<f64>,
    w1: &mut Array2<f64>, b1: &mut Array1<f64>,
    w2: &mut Array2<f64>, b2: &mut Array1<f64>,
    lr: f64,
) {
    let (z1, a1, _z2, a2) = model(x, w1, b1, w2, b2);

    let dz2 = &a2 - y;
    let dw2 = a1.view().insert_axis(Axis(1)).dot(&dz2.view().insert_axis(Axis(0)));
    let db2 = dz2.clone();

    let dz1 = w2.dot(&dz2) * z1.mapv(sigmoid_derivative);
    let dw1 = x.view().insert_axis(Axis(1)).dot(&dz1.view().insert_axis(Axis(0)));
    let db1 = dz1.clone();

    *w2 -= &(dw2 * lr);
    *b2 -= &(db2 * lr);
    *w1 -= &(dw1 * lr);
    *b1 -= &(db1 * lr);
}

fn gradient_descent(
    data: &Vec<([f64; 2], [f64; 2])>,
    w1: &mut Array2<f64>, b1: &mut Array1<f64>,
    w2: &mut Array2<f64>, b2: &mut Array1<f64>,
    lr: f64,
    epochs: usize,
) {
    for epoch in 0..epochs {
        for (x_raw, y_raw) in data {
            let x = Array1::from_vec(x_raw.to_vec());
            let y = Array1::from_vec(y_raw.to_vec());
            train_step(&x, &y, w1, b1, w2, b2, lr);
        }

        if epoch % 1000 == 0 {
            let mut total_loss = 0.0;
            for (x_raw, y_raw) in data {
                let x = Array1::from_vec(x_raw.to_vec());
                let y = Array1::from_vec(y_raw.to_vec());
                let (_, _, _, output) = model(&x, w1, b1, w2, b2);
                total_loss += binary_cross_entropy(&output, &y);
            }
            println!("Epoch {epoch}, Total Loss: {:.4}", total_loss);
        }
    }
}

fn accuracy(
    data: &Vec<([f64; 2], [f64; 2])>,
    w1: &Array2<f64>, b1: &Array1<f64>,
    w2: &Array2<f64>, b2: &Array1<f64>,
) -> f64 {
    let mut correct = 0;

    for (x_raw, y_raw) in data {
        let x = Array1::from_vec(x_raw.to_vec());
        let y = Array1::from_vec(y_raw.to_vec());

        let (_, _, _, out) = model(&x, w1, b1, w2, b2);
        let predicted = out.mapv(|v| if v > 0.5 { 1.0 } else { 0.0 });

        if predicted == y {
            correct += 1;
        }
    }

    correct as f64 / data.len() as f64
}


fn main() {
    let data = vec![
        ([0.0, 0.0], [0.0, 0.0]),
        ([0.0, 1.0], [0.0, 1.0]),
        ([1.0, 0.0], [1.0, 1.0]),
        ([1.0, 1.0], [1.0, 0.0]),
    ];

    let mut w1 = Array2::<f64>::random((2, 4), Uniform::new(-1.0, 1.0));
    let mut b1 = Array1::<f64>::zeros(4);
    let mut w2 = Array2::<f64>::random((4, 2), Uniform::new(-1.0, 1.0));
    let mut b2 = Array1::<f64>::zeros(2);

    let learning_rate = 0.1;
    let epochs = 10000;

    gradient_descent(&data, &mut w1, &mut b1, &mut w2, &mut b2, learning_rate, epochs);

    println!("\nFinal Predictions:");
    for (x_raw, _) in &data {
        let x = Array1::from_vec(x_raw.to_vec());
        let (_, _, _, out) = model(&x, &w1, &b1, &w2, &b2);
        let rounded = out.mapv(|v| if v > 0.5 { 1.0 } else { 0.0 });
        println!("Input: {:?} → Output: {:?}", x_raw, rounded.to_vec());
    }

    let acc = accuracy(&data, &w1, &b1, &w2, &b2);
    println!("Model accuracy: {:.2}%", acc * 100.0);

}
