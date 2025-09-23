use candle_core::{D, DType, Device, Module, Tensor};
use candle_nn::{VarBuilder, VarMap, linear_no_bias};

fn cdist(x1: &Tensor, x2: &Tensor) -> Result<Tensor, Box<dyn std::error::Error>> {
    let x1 = x1.unsqueeze(0)?;
    let x2 = x2.unsqueeze(1)?;
    Ok(x1
        .broadcast_sub(&x2)?
        .sqr()?
        .sum(D::Minus1)?
        .sqrt()?
        .transpose(D::Minus1, D::Minus2)?)
}

fn zscore_normalize(data: &Tensor) -> Result<Tensor, Box<dyn std::error::Error>> {
    let mean = data.mean(0)?;
    let squared_diff = data.broadcast_sub(&mean)?.sqr()?;
    let variance = squared_diff.mean(0)?;
    let std_dev = variance.sqrt()?;
    let normalized = data.broadcast_sub(&mean)?.broadcast_div(&std_dev)?;
    Ok(normalized)
}

fn cov(data: &Tensor, device: &Device) -> Result<Tensor, Box<dyn std::error::Error>> {
    let mean = data.mean(0)?;
    let centered = data.broadcast_sub(&mean)?;
    let (m, _) = data.shape().dims2()?;
    let cov = centered
        .transpose(D::Minus1, D::Minus2)?
        .matmul(&centered)?
        .broadcast_div(&Tensor::new((m - 1) as f64, device)?)?;
    Ok(cov)
}

#[allow(dead_code)]
fn scalar() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::new_cuda(0)?;
    let scalar_tensor = Tensor::new(std::f32::consts::PI, &device)?;
    let scalar = scalar_tensor.to_scalar::<f32>()?;
    println!("{scalar}");
    println!("ndim {:?}", scalar_tensor.dims().len());
    println!("shape {:?}", scalar_tensor.shape());
    Ok(())
}

#[allow(dead_code)]
fn vector() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::new_cuda(0)?;
    let vector = vec![1., 2., 3., 4.];
    let vector_tensor = Tensor::from_slice(&vector, (4,), &device)?;
    println!("{vector_tensor}");
    println!("ndim {:?}", vector_tensor.dims().len());
    println!("shape {:?}", vector_tensor.shape());
    Ok(())
}

#[allow(dead_code)]
fn info() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::new_cuda(0)?;
    let tensor = Tensor::rand(0f32, 1., (2, 4), &device)?;

    println!("Tensor Shape {:?}", tensor.shape());
    println!("Tensor DType {:?}", tensor.dtype());
    println!("Tensor Device {:?}", tensor.device());

    Ok(())
}

#[allow(dead_code)]
fn concat() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::new_cuda(0)?;
    let tensor1 = Tensor::ones((4, 4), DType::F32, &device)?;
    let tensor2 = Tensor::zeros((4, 4), DType::F32, &device)?;
    let tensor = Tensor::cat(&[&tensor1, &tensor2], 0)?;
    println!("Tensor {tensor}");
    Ok(())
}

#[allow(dead_code)]
fn multiple() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::new_cuda(0)?;
    let tensor1 = Tensor::rand(0f32, 1., (4, 4), &device)?;
    let tensor2 = Tensor::rand(0f32, 1., (4, 4), &device)?;
    let tensor = Tensor::mul(&tensor1, &tensor2)?;
    println!("Tensor {tensor}");
    Ok(())
}

#[allow(dead_code)]
fn range() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::new_cuda(0)?;

    let tensor1 = Tensor::arange(0f32, 6f32, &device)?;
    println!("Tensor {tensor1}");

    let tensor1 = Tensor::arange(0f32, 6f32, &device)?.reshape((2, 3))?;
    println!("Tensor {tensor1}");

    let tensor2 = Tensor::arange(0f32, 12f32, &device)?.reshape((3, 4))?;
    println!("Tensor {tensor2}");

    let tensor = tensor1.matmul(&tensor2)?;
    println!("Tensor {tensor}");
    Ok(())
}

#[allow(dead_code)]
fn broadcast_add() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::new_cuda(0)?;

    let tensor1 = Tensor::arange(0f32, 6f32, &device)?.reshape((2, 3))?;
    println!("Tensor {tensor1}");

    let tensor = tensor1.broadcast_add(&Tensor::new(10f32, &device)?)?;
    println!("Tensor {tensor}");

    Ok(())
}

#[allow(dead_code)]
fn broadcast_mul() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::new_cuda(0)?;

    let tensor1 = Tensor::arange(0f32, 6f32, &device)?.reshape((2, 3))?;
    println!("Tensor {tensor1}");

    let tensor = tensor1.broadcast_mul(&Tensor::new(10f32, &device)?)?;
    println!("Tensor {tensor}");

    Ok(())
}

#[allow(dead_code)]
fn indexing() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::new_cuda(0)?;

    let rand_tensor = Tensor::rand(0f32, 1., (20, 1), &device)?;
    println!("Tensor {rand_tensor}");

    let index_tensor = Tensor::new(&[0u32, 3u32], &device)?;
    println!("Index tensor {index_tensor}");

    let selected = rand_tensor.index_select(&index_tensor, 0)?;
    println!("Selected tensor {selected}");
    Ok(())
}

#[allow(dead_code)]
fn argmax() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::new_cuda(0)?;

    let tensor = Tensor::arange(0f32, 10f32, &device)?;
    println!("Tensor {tensor}");

    let argmax = tensor.argmax(0)?;
    println!("Argmax {argmax}");

    let tensor = Tensor::rand(0f32, 1., (3, 3), &device)?;
    println!("Tensor {tensor}");

    let argmax = tensor.argmax(0)?;
    println!("Argmax: dim 0  {argmax}");

    let argmax = tensor.argmax(1)?;
    println!("Argmax: dim 1  {argmax}");

    Ok(())
}

#[allow(dead_code)]
fn cdist_test() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::new_cuda(0)?;

    let data1 = vec![0.9041, 0.0196, -0.3108, -2.4423, -0.4821, 1.059];
    let tensor1 = Tensor::from_slice(&data1, (3, 2), &device)?;

    let data2 = vec![-2.1763, -0.4713, -0.6986, 1.3702];
    let tensor2 = Tensor::from_slice(&data2, (2, 2), &device)?;

    let tensor = cdist(&tensor1, &tensor2)?;
    println!("Tensor {tensor}");

    Ok(())
}

#[allow(dead_code)]
fn zscore_normalize_test() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::new_cuda(0)?;

    let data = vec![3., 2., 15., 6., 0., 10., 1., 18.];
    let tensor = Tensor::from_slice(&data, (4, 2), &device)?;
    println!("Tensor {tensor}");
    let tensor = zscore_normalize(&tensor)?;
    println!("Normalized Tensor {tensor}");

    Ok(())
}

#[allow(dead_code)]
fn cov_test() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::new_cuda(0)?;

    let data = vec![80., 70., 63., 20., 100., 50.];
    let tensor = Tensor::from_slice(&data, (3, 2), &device)?;
    println!("Tensor {tensor}");
    let tensor = cov(&tensor, &device)?;
    println!("Cov Tensor {tensor}");

    let data = vec![68., 29., 60., 26., 58., 30., 40., 35.];
    let tensor = Tensor::from_slice(&data, (4, 2), &device)?;
    println!("Tensor {tensor}");
    let tensor = cov(&tensor, &device)?;
    println!("Cov Tensor {tensor}");

    let data = vec![80., 70., 80., 63., 20., 30., 100., 50., 50.];
    let tensor = Tensor::from_slice(&data, (3, 3), &device)?;
    println!("Tensor {tensor}");
    let tensor = cov(&tensor, &device)?;
    println!("Cov Tensor {tensor}");

    Ok(())
}

#[allow(dead_code)]
fn r_square_test() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::new_cuda(0)?;

    let y_true = vec![3., -0.5, 2., 7.];
    let y_pred = vec![2.5, 0.0, 2., 8.];

    let y_true = Tensor::from_slice(&y_true, (4,), &device)?;
    let y_pred = Tensor::from_slice(&y_pred, (4,), &device)?;

    let y_mean = y_true.mean(0).unwrap();
    let ss_tot = y_true
        .broadcast_sub(&y_mean)
        .unwrap()
        .sqr()
        .unwrap()
        .sum(D::Minus1)
        .unwrap();
    let ss_res = y_true
        .broadcast_sub(&y_pred)
        .unwrap()
        .sqr()
        .unwrap()
        .sum(D::Minus1)
        .unwrap();
    let r_square = ss_res.broadcast_div(&ss_tot).unwrap();
    println!("R Square: {:?}", r_square);
    Ok(())
}

#[allow(dead_code)]
fn variance_test() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::new_cuda(0)?;

    let data = vec![2., 4., 6., 1., 3., 5., 3., 6., 9., 4., 8., 12.];
    let data = Tensor::from_slice(&data, (4, 3), &device)?;

    let rows = data.shape().dims2()?.0;
    for row in 0..rows {
        let row_tensor = data
            .index_select(&Tensor::new(&[row as u32], &device)?, 0)?
            .squeeze(0)?;
        println!("row_tensor {row_tensor}");
    }

    Ok(())
}

#[allow(dead_code)]
fn rmsnorm_test() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::new_cuda(0)?;

    let data = Tensor::rand(0f32, 1., (2, 3, 4), &device)?;
    let x = data;
    let x_dtype = x.dtype();
    let internal_dtype = match x_dtype {
        DType::F16 | DType::BF16 => DType::F32,
        d => d,
    };
    let hidden_size = x.dim(D::Minus1)?;
    let x = x.to_dtype(internal_dtype)?;
    let norm_x = (x.sqr()?.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
    let x_normed = x.broadcast_div(&(norm_x).sqrt()?)?;
    println!("X {x}");
    println!("Normed x {x_normed}");

    Ok(())
}

#[allow(dead_code)]
fn tril_test() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::new_cuda(0)?;
    let tril = Tensor::tril2(8, DType::F32, &device)?;
    println!("Tril {}", tril);

    let sum = tril.sum(D::Minus1)?;
    let sum = sum.unsqueeze(1)?;
    println!("Sum {}", sum);

    let normalized_tril = tril.broadcast_div(&sum)?;
    println!("Normalized Tril {}", normalized_tril);

    Ok(())
}

#[allow(dead_code)]
fn attention_mask(size: usize) -> Result<Tensor, Box<dyn std::error::Error>> {
    let device = Device::new_cuda(0)?;
    let mask: Vec<f32> = (0..size)
        .flat_map(|i| (0..size).map(move |j| if i < j { f32::NEG_INFINITY } else { 0f32 }))
        .collect();
    let mask = Tensor::from_slice(&mask, (size, size), &device)?;
    Ok(mask)
}

#[allow(dead_code)]
fn softmax_test() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::new_cuda(0)?;
    let wei = Tensor::rand(0f32, 1., (8, 8), &device)?;
    let mask = attention_mask(8)?;
    let wei = wei.broadcast_add(&mask)?;
    let wei = candle_nn::ops::softmax(&wei, D::Minus1)?;
    println!("Softmax Wei {}", wei);
    Ok(())
}
#[allow(dead_code)]
fn reshape_test() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::new_cuda(0)?;
    let tensor = Tensor::rand(0f32, 1., (2, 4, 4), &device)?;
    println!("Tensor {}", tensor);
    let tensor = tensor.reshape((8, 4))?;
    println!("Squeezed Tensor {}", tensor);
    Ok(())
}

#[allow(dead_code)]
fn attention_test() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::new_cuda(0)?;

    let B = 4;
    let T = 8;
    let d = 16;
    let n_heads = 1;
    let d_k = d / n_heads;
    let V = 100;

    let X = Tensor::rand(0f32, 1., (B, T, d), &device)?;
    println!("X {:?}", X);

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let W_Q = linear_no_bias(d, d_k, vb.pp("W_Q"))?;
    let W_K = linear_no_bias(d, d_k, vb.pp("W_K"))?;
    let W_V = linear_no_bias(d, d_k, vb.pp("W_V"))?;

    println!("W_Q {:?}", W_Q);
    println!("W_K {:?}", W_K);
    println!("W_V {:?}", W_V);

    let q = W_Q.forward(&X)?;
    let k = W_K.forward(&X)?;

    let wei = (q.matmul(&k.transpose(D::Minus1, D::Minus2)?)? / (d_k as f64).sqrt())?;
    let mask = attention_mask(T)?;
    let wei = wei.broadcast_add(&mask)?;
    let wei = candle_nn::ops::softmax(&wei, D::Minus1)?;
    println!("Wei {}", wei);

    let v = W_V.forward(&X)?;
    let out = wei.matmul(&v)?;
    println!("Out {:?}", out);
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    rmsnorm_test()?;
    Ok(())
}
