use dfdx::{data::ExactSizeDataset, optim::Sgd, prelude::*};

type MlpConfig = (
    Linear<5, 32>,
    ReLU,
    Linear<32, 32>,
    ReLU,
    Linear<32, 2>,
    Tanh,
);

const BATCH_SIZE: usize = 32;

fn main() {
    dfdx::flush_denormals_to_zero();
    let device = AutoDevice::default();
    let mut mlp = device.build_module::<MlpConfig, f32>();
    let mut grads = mlp.alloc_grads();

    let dataset = load_dataset();
    
    // let opt = Sgd::new(
    //     &mlp,
    //     SgdConfig {
    //         lr: 1e-3,
    //         momentum: Some(Momentum::Nesterov(0.9)),
    //         weight_decay: None,
    //     },
    // );
}

pub struct Data {
    vals: Vec<Tensor<Rank2<100,100>, f32, Cpu, NoneTape>>
}

impl ExactSizeDataset for Data {
    type Item<'a> = Tensor<Rank2<100, 100>, f32, Cpu, NoneTape>
    where
        Self: 'a;

    fn get(&self, index: usize) -> Self::Item<'_> {
        self.vals.get(index).unwrap().clone()
    }

    fn len(&self) -> usize {
        self.vals.len()
    }
}

impl Data {
    fn new() -> Self {
        Self { vals: vec![] }
    }

    fn from_vec(v: Vec<Tensor<Rank2<100,100>,f32,Cpu,NoneTape>>) -> Self {
        Self { vals: v }
    }
}

fn load_dataset() -> Data {
    let device = AutoDevice::default();
    let mut ret = vec![];

    for i in 1..=10 {
        println!("Processing k{i}");
        let gray = image::open(format!("src/data/k1")).unwrap().to_luma8();
        gray.save(format!("src/data_updated/k{i}")).unwrap();
    }

    Data::from_vec(ret)
}
