use anyhow::Result;

use image::imageops;
use image::GrayAlphaImage;

use std::env;
use std::ffi::OsStr;
use std::fs::{read_dir, File};
use std::ops::{Add, Div};
use tch::kind::{FLOAT_CPU, FLOAT_CUDA};
use tch::nn::{Conv2D, ModuleT, OptimizerConfig};
use tch::vision::dataset::Dataset;
use tch::{nn, CModule, Cuda, Device, IndexOp, Kind, Tensor};

mod macros;
pub use macros::*;

//noinspection SpellCheckingInspection
fn main() {
    let arg = env::args().nth(1).unwrap();
    match arg.as_str() {
        "genstrokes" => generate_strokes(),
        "generate" => generate_dataset(),
        "train" => match train() {
            Ok(_) => println!("Training complete"),
            Err(e) => eprintln!("Error: {e}"),
        },
        "predict_stroke" => predict_stroke(),
        _ => {
            // TODO: Add more options
            println!("Usage: gregg_anni_interpreter [genstrokes|generate|train]");
        }
    }
}

// Layout of the neural network: 1 input layer, 1 hidden layer, 1 output layer.
// FC stands for fully connected.
#[derive(Debug)]
pub struct InitialIdentificationCnn {
    pub conv1: Conv2D,
    pub conv2: Conv2D,
    pub fc1: nn::Linear,
    pub fc2: nn::Linear,
}

impl InitialIdentificationCnn {
    pub fn new(vs: &nn::Path) -> InitialIdentificationCnn {
        let conv1 = nn::conv2d(vs, 3, 32, 2, Default::default());
        let conv2 = nn::conv2d(vs, 32, 64, 2, Default::default());
        let fc1 = nn::linear(vs, 64 * (BATCH_SIZE as i64), 1280, Default::default());
        let fc2 = nn::linear(vs, 1280, BATCH_SIZE as i64, Default::default());
        InitialIdentificationCnn {
            conv1,
            conv2,
            fc1,
            fc2,
        }
    }
}

impl ModuleT for InitialIdentificationCnn {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        // Add the new 3 dimensions to the input
        let proper_xs = xs.unsqueeze(1).unsqueeze(2).unsqueeze(3);
        // Reshape the input to the correct shape
        let proper_xs = proper_xs.resize(&[BATCH_SIZE as i64, 3, 10, 10]);

        // Change the type of the input to f32
        let proper_xs = proper_xs.to_kind(Kind::Float);

        // Expects input of shape [batch_size, channels, height, width]
        // In this case, batch_size = 6, channels = 3, height = 10, width = 10
        let proper_xs = proper_xs.view([BATCH_SIZE as i64, 3, 10, 10]);

        let proper_xs = proper_xs.apply(&self.conv1);
        let proper_xs = proper_xs.max_pool2d_default(2);
        let proper_xs = proper_xs.apply(&self.conv2);
        let proper_xs = proper_xs.max_pool2d_default(2);
        let proper_xs = proper_xs.view([-1, 64 * (BATCH_SIZE as i64)]);
        let proper_xs = proper_xs.apply(&self.fc1);
        let proper_xs = proper_xs.relu();
        let proper_xs = proper_xs.dropout(0.5, train);
        let proper_xs = proper_xs.apply(&self.fc2);
        if proper_xs.size() != [1, BATCH_SIZE as i64] {
            panic!("Shape of output is not correct");
        } else {
            proper_xs.squeeze()
        }
    }
}

fn predict_stroke() {
    // Check if src/data/model.pt exists
    let model_path = "src/data/model.pt";
    if !std::path::Path::new(model_path).exists() {
        println!("Model not found. Please train the model first.");
        return;
    }

    // Check if argument img is provided
    let img_path = env::args().nth(2).unwrap();
    if !std::path::Path::new(&img_path).exists() {
        println!("Image not found. Please provide a valid image path.");
        return;
    }

    // We want to use the GPU whenever possible
    let device = Device::cuda_if_available();

    // Load the data
    let image = tch::vision::imagenet::load_image(img_path).unwrap();
    // Pad
    // Find the average color of the pixels in the image, outside of those with a brightness <.5
    let mut sum = Tensor::zeros(&[1], (Kind::Float, Device::cuda_if_available()));
    let mut count = 0;
    for y in 0..image.size()[1] {
        for x in 0..image.size()[2] {
            let pixel_r = image.i((0, y, x));
            let pixel_g = image.i((1, y, x));
            let pixel_b = image.i((2, y, x));

            let pixel = Tensor::stack(&[pixel_r, pixel_g, pixel_b], 0);

            let pixel = pixel.sum(Kind::Float).div(3f64);

            let pixel_lt = pixel.lt(0.5);
            if !pixel_lt.size().is_empty() {
                panic!("Pixel_lt size is not 1/0! It is {:?}", pixel_lt.size());
            }
            // SAFETY: We know that pixel_lt is a tensor of size 0 (or maybe 1?), due to the
            // check above
            if bool::from(pixel_lt) {
                continue;
            }

            sum = sum.add(pixel);
            count += 1;
        }
    }

    let avg = sum.div(count as f64);

    // Invert the indices since fraction / reciprocal = 1
    let left_pad = (100 - image.size()[2]) / 2; // Divide by two so half goes to right side
    let right_pad = 100 - image.size()[2] - left_pad;
    let top_pad = (100 - image.size()[1]) / 2; // Divide by two so half goes to bottom
    let bottom_pad = 100 - image.size()[1] - top_pad;

    let image = image.pad(
        &[left_pad, right_pad, top_pad, bottom_pad],
        "constant",
        f64::from(avg),
    );
    // Resize
    let image = tch::vision::image::resize_preserve_aspect_ratio(&image, 100, 100).unwrap();
    let image = image.to_kind(Kind::Float).to_device(device);
    let mut model = tch::CModule::load(model_path).unwrap();
    model.set_eval();
    let output = model.forward_ts(&[image]).unwrap();
    let output = output.softmax(-1, Kind::Float);
    let iter = output.iter::<f64>().unwrap();
    let mut real_output = vec![];
    for output in iter {
        real_output.push(output)
    }
    println!("{real_output:?}");

}

pub const BATCH_SIZE: usize = 10;

// Train the model from the dataset (src/data/*.png)
// Implemented using tch-rs
fn train() -> Result<()> {
    let m = load_initial_dataset();
    let mut vs = nn::VarStore::new(Device::cuda_if_available());
    let net = InitialIdentificationCnn::new(&vs.root());
    let mut opt = nn::Adam::default().build(&vs, 1e-4)?;
    let mut accuracy = 0f64;
    let mut accuracy_count = 0f64;
    let initial_time = std::time::Instant::now();
    for epoch in 1..=100 {
        for (bimages, blabels) in m
            .train_iter(BATCH_SIZE as i64)
            .shuffle()
            .to_device(vs.device())
        {
            // Convert the input type from u8 to f32
            let bimages = bimages.to_kind(Kind::Float);
            // Bring the input into the range [0, 1]
            let bimages = bimages.div(255.0);
            let loss = net
                .forward_t(&bimages, true)
                .cross_entropy_for_logits(&blabels);
            println!("Epoch: {epoch}, Loss: {}", i64::from(loss.shallow_clone()));
            opt.backward_step(&loss);
        }
        // Compute the accuracy on the test set, using batches of size 6.
        let test_accuracy = net.batch_accuracy_for_logits(
            &m.test_images,
            &m.test_labels,
            vs.device(),
            BATCH_SIZE as i64,
        );
        println!("Epoch: {epoch}, Test accuracy: {test_accuracy}");
        accuracy += test_accuracy;
        accuracy_count += 1f64;
    }
    vs.freeze();

    println!("Overall accuracy: {}", accuracy / accuracy_count);

    println!("Saving model...");
    // Save the model
    let mut closure = |input: &[Tensor]| vec![net.forward_t(&input[0], false)];
    let model = CModule::create_by_tracing(
        "MyModule",
        "forward",
        &[Tensor::zeros(
            &[784],
            if Cuda::is_available() {
                FLOAT_CUDA
            } else {
                FLOAT_CPU
            },
        )],
        &mut closure,
    )?;
    model.save("src/data/model.pt")?;

    println!("Model saved");

    let time_since_start = initial_time.elapsed().as_secs_f32();
    println!("Time took: {time_since_start:.2}s");

    Ok(())
}

fn load_initial_dataset() -> Dataset {
    let mut data: Vec<Tensor> = vec![];

    for img_direntry in read_dir("src/data/").expect("Could not read directory") {
        let img_direntry = img_direntry.unwrap().path();

        if img_direntry.extension() != Some(OsStr::new("png")) {
            continue;
        }

        let mut img = tch::vision::image::load(img_direntry.as_path())
            .expect("Could not load image to tensor.");

        img = img.to_kind(Kind::Float);

        // Find the average color of the pixels in the image, outside of those with a brightness <.5
        let mut sum = Tensor::zeros(&[1], (Kind::Float, Device::cuda_if_available()));
        let mut count = 0;
        for y in 0..img.size()[1] {
            for x in 0..img.size()[2] {
                let pixel_r = img.i((0, y, x));
                let pixel_g = img.i((1, y, x));
                let pixel_b = img.i((2, y, x));

                let pixel = Tensor::stack(&[pixel_r, pixel_g, pixel_b], 0);

                let pixel = pixel.sum(Kind::Float).div(3f64);

                let pixel_lt = pixel.lt(0.5);
                if !pixel_lt.size().is_empty() {
                    panic!("Pixel_lt size is not 1/0! It is {:?}", pixel_lt.size());
                }
                // SAFETY: We know that pixel_lt is a tensor of size 0 (or maybe 1?), due to the
                // check above
                if bool::from(pixel_lt) {
                    continue;
                }

                sum = sum.add(pixel);
                count += 1;
            }
        }

        let avg = sum.div(count as f64);

        // Invert the indices since fraction / reciprocal = 1
        let left_pad = (100 - img.size()[2]) / 2; // Divide by two so half goes to right side
        let right_pad = 100 - img.size()[2] - left_pad;
        let top_pad = (100 - img.size()[1]) / 2; // Divide by two so half goes to bottom
        let bottom_pad = 100 - img.size()[1] - top_pad;

        let img = img.pad(
            &[left_pad, right_pad, top_pad, bottom_pad],
            "constant",
            f64::from(avg),
        );

        // Convert the image to grayscale
        let mut img = tensor_to_grayscale(&img);

        tch::vision::image::save(
            &img,
            format!(
                "src/data/gray/{}",
                img_direntry.file_name().unwrap().to_string_lossy()
            ),
        )
        .expect("Could not save image");

        // If necessary, change the first dimension's size to 3 (the number of channels in RGB)
        if img.size()[0] == 1 {
            // Necessary.
            let mut img_3 =
                Tensor::zeros(&[3, 100, 100], (Kind::Float, Device::cuda_if_available()));
            for i in 0..3 {
                img_3.copy_(&img.i((i, .., ..)));
            }
            img = img_3;
        }
        data.push(img.shallow_clone());
    }
    // Divide it into 90% training and 10% testing
    let (train_images, test_images) = data.split_at((data.len() as f64 * 0.9).round() as usize);
    // Convert it from a slice (&[Tensor]) to a plain Tensor
    let train_images = Tensor::stack(train_images, 0);
    let test_images = Tensor::stack(test_images, 0);

    let mut labels: Vec<u8> = vec![];
    for _ in 0..(train_images.size()[0] + test_images.size()[0]) {
        labels.push(0);
    }

    // Split the labels into 90% training and 10% testing
    let (train_labels, test_labels) = labels.split_at((labels.len() as f64 * 0.9).round() as usize);

    // Convert it from a slice (&[u8]) to a plain Tensor
    let train_labels = Tensor::of_slice(train_labels);
    let test_labels = Tensor::of_slice(test_labels);

    let train_labels_len = train_labels.size()[0];
    let test_labels_len = test_labels.size()[0];

    Dataset {
        train_images,
        train_labels,
        test_images,
        test_labels,
        labels: train_labels_len + test_labels_len,
    }
}

#[inline(never)]
fn tensor_to_grayscale(tensor: &Tensor) -> Tensor {
    // Iterate over all of the pixels, as 1D tensors

    let tensor_noref = tensor.to_kind(Kind::Uint8);
    let tensor = &tensor_noref;
    for y in 0..tensor.size()[1] {
        for x in 0..tensor.size()[2] {
            let pixel_r = tensor.i((0, y, x));
            let pixel_g = tensor.i((1, y, x));
            let pixel_b = tensor.i((2, y, x));

            let pixel_color = Tensor::stack(&[pixel_r, pixel_g, pixel_b], 0);

            let pixel_color = pixel_color
                .sum(Kind::Float)
                .div(3f64)
                .round_()
                .to_kind(Kind::Uint8);

            // Expand the grayscale pixel to 3 channels, so it can be used as a color
            let pixel_color = pixel_color.expand(&[3, 1, 1], false);

            let mut tensor_ref = tensor.put(&Tensor::of_slice(&[0, y, x]), &pixel_color, false);
            tensor_ref = tensor_ref.put(&Tensor::of_slice(&[1, y, x]), &pixel_color, false);
            tensor_ref = tensor_ref.put(&Tensor::of_slice(&[2, y, x]), &pixel_color, false);
            let tensor = &tensor_ref;
        }
    }

    tensor.shallow_clone()
}

/// Simply flips a few images vertically to save time in drawing them, as well as expanding the
/// images to 2x2 pixels (padding with 0s)
/// This should be run only once after each time the images change.
/// Generates:
/// - src/data/r1.png
/// - src/data/r2.png
/// - src/data/r3.png
/// - src/data/r4.png
/// - src/data/r5.png
/// - src/data/r6.png
/// - src/data/r7.png
/// - src/data/r8.png
/// - src/data/r9.png
/// - src/data/r10.png
/// from src/data/1.png, src/data/2.png, ..., src/data/10.png (the K-strokes)
///
/// And generates
/// - src/data/l1.png
/// - src/data/l2.png
/// - src/data/l3.png
/// - src/data/l4.png
/// - src/data/l5.png
/// - src/data/l6.png
/// - src/data/l7.png
/// - src/data/l8.png
/// - src/data/l9.png
/// - src/data/l10.png
/// from src/data/g1.png, src/data/g2.png, ..., src/data/g10.png (the G-strokes)
fn generate_strokes() {
    for i in 1..=10 {
        println!("Generating stroke k{i}");
        let img = image::open(format!("src/data/k{i}.png")).unwrap().to_rgb8();

        // Up the contrast by 50%
        let new_img = imageops::contrast(&img, 50f32);

        // Save the modified image
        new_img
            .save(format!("src/data/k{i}.png"))
            .expect("Could not save image");

        let flipped = imageops::flip_vertical(&new_img);
        flipped
            .save(format!("src/data/r{i}.png"))
            .expect("Could not save image")
    }
    for i in 1..=10 {
        let img = image::open(format!("src/data/g{i}.png")).unwrap();
        let flipped = imageops::flip_vertical(&img);
        flipped.save(format!("src/data/l{i}.png")).unwrap();
    }
}

const IMAGE_PATHS: [&str; 20] = [
    "src/data/k1.png",
    "src/data/k2.png",
    "src/data/k3.png",
    "src/data/k4.png",
    "src/data/k5.png",
    "src/data/k6.png",
    "src/data/k7.png",
    "src/data/k8.png",
    "src/data/k9.png",
    "src/data/k10.png",
    "src/data/r1.png",
    "src/data/r2.png",
    "src/data/r3.png",
    "src/data/r4.png",
    "src/data/r5.png",
    "src/data/r6.png",
    "src/data/r7.png",
    "src/data/r8.png",
    "src/data/r9.png",
    "src/data/r10.png",
];

/// Generate the initial dataset as a csv (from the image)
///
/// Will output the dataset to the file `data/data_as.csv`
fn generate_dataset() {
    match std::fs::create_dir_all("src/data") {
        Ok(_) => {}
        Err(e) => {
            panic!("Error creating directory: {e}");
        }
    }

    // Create the file, if it doesn't exist
    let file = match File::create("src/data/data_as.csv") {
        Ok(f) => f,
        Err(e) => {
            panic!("Error creating file: {e}");
        }
    };

    let mut writer = csv::Writer::from_writer(file);
    let mut reader = csv::Reader::from_path("src/data/data_as.csv").unwrap();
    // If the file is empty, write the header
    if reader.records().count() == 0 {
        writer
            .write_record(
                [
                    "img",
                    "x",
                    "y",
                    "brightness",
                    "alpha",
                    "surrounding_brightness",
                ]
                .iter(),
            )
            .unwrap();
    }

    // Open the images
    for img_str in IMAGE_PATHS.iter() {
        println!("Processing image: {img_str}");
        let img = image::open(img_str).unwrap();
        // Luma8 is an 8-bit grayscale image
        let img = img.to_luma_alpha8();
        // Iterate over the rows (aka y coordinate)
        for y in 0..(img.height() - 1) {
            // Iterate over the columns (aka x coordinate)
            for x in 0..(img.width() - 1) {
                println!("Processing pixel: {img_str}:({x}, {y})");
                // Current pixel
                // The LumaA8 is represented as [brightness, alpha], of type u8.
                let pixel = img.get_pixel(x, y);
                // Current pixel's alpha (transparency) value
                let alpha = pixel[1];
                // Current pixel's brightness value
                let brightness = pixel[0];

                let surrounding_brightness = get_surrounding_brightness(&img, x, y);

                // Get whether or not the pixel is identified as a drawn pixel
                let is_pixel_good = pixel_test(brightness, alpha, surrounding_brightness);

                // If the pixel isn't good, short-circuit (aka skip to the next pixel)
                if !is_pixel_good {
                    continue;
                }

                // Write the pixel to the csv
                writer
                    .write_record(
                        [
                            img_str.to_string(),
                            x.to_string(),
                            y.to_string(),
                            brightness.to_string(),
                            alpha.to_string(),
                            surrounding_brightness.to_string(),
                        ]
                        .iter(),
                    )
                    .unwrap();
            }
        }
    }
    // Flush the writer (to guarantee the data is written)
    writer.flush().unwrap();

    println!("Dataset generated");
}

fn get_surrounding_brightness(img: &GrayAlphaImage, x: u32, y: u32) -> u8 {
    // Brightness of the pixels surrounding the current pixel, averaged (mean)
    let mut surrounding_brightness: u32 = 0;
    // Pixels that have been counted so far
    let mut surrounding_pixels_counted = 0;
    // Alias each pixel to a variable, for readability
    // We also must use `checked_sub` for left and top pixels, because they may be out of bounds
    let tpl = img.get_pixel(x.saturating_sub(1), y.saturating_sub(1));
    let top = img.get_pixel(x, y.saturating_sub(1));
    let tpr = img.get_pixel(x + 1, y.saturating_sub(1));
    let lft = img.get_pixel(x.saturating_sub(1), y);
    let rht = img.get_pixel(x + 1, y);
    let btl = img.get_pixel(x.saturating_sub(1), y + 1);
    let bot = img.get_pixel(x, y + 1);
    let btr = img.get_pixel(x + 1, y + 1);

    #[rustfmt::skip]
    let surrounding_pixels = [
        tpl,  top,  tpr,
        lft,/*cur*/rht,
        btl,  bot,  btr,
    ];

    for pixel in surrounding_pixels.iter() {
        // If the pixel is transparent, don't count it
        if pixel[1] == 0 {
            surrounding_pixels_counted += 1;
            continue;
        }
        // Add the brightness to the total
        surrounding_brightness += pixel[0] as u32;
        // Increment the number of pixels that have been counted
        surrounding_pixels_counted += 1;
    }

    // Take the average of the surrounding pixels
    // `checked_div` is used to prevent panicking by divide by zero
    surrounding_brightness
        .checked_div(surrounding_pixels_counted)
        .unwrap_or(0) as u8
}

fn pixel_test(brightness: u8, alpha: u8, surrounding_brightness: u8) -> bool {
    // If the pixel is transparent, it's not good
    if alpha == 0 {
        return false;
    }

    // If pixel is sufficiently dark compared to the surrounding pixels, it's considered "good" (aka
    // a drawn pixel)
    // Sufficiency is determined by the `BRIGHTNESS_THRESHOLD` environment variable.
    // If the environment variable is not set, it defaults to 20.
    if brightness
        < env::var("BRIGHTNESS_THRESHOLD")
            .unwrap_or("2".to_string())
            .parse::<u8>()
            .expect("BRIGHTNESS_THRESHOLD must be an 8-bit unsigned integer")
            .saturating_sub(surrounding_brightness)
    {
        return true;
    }

    // Otherwise, it's not good
    false
}
