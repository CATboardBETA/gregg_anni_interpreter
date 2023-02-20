use csv::Reader;
use image::{imageops, GrayAlphaImage};

use ndarray::Array2;

use ndarray::Axis;
use std::env;
use std::fs::File;
use tch::{nn, Device};

mod macros;
pub use macros::*;

type InitialDataset = (Array2<String>, Array2<u8>);

//noinspection SpellCheckingInspection
fn main() {
    let arg = env::args().nth(1).unwrap();
    match arg.as_str() {
        "genstrokes" => generate_strokes(),
        "generate" => generate_dataset(),
        "train" => train(),
        _ => {
            // TODO: Add more options
            println!("Usage: gregg_anni_interpreter [genstrokes|generate|train]");
        }
    }
}

// // Layout of the neural network: 1 input layer, 1 hidden layer, 1 output layer. Input layer has
// pub struct InitialIdentificationCnn {
//     conv1
// }

// Train the model from the dataset (src/data/data_as.csv)
// Implemented using tch-rs
fn train() {
    let mut reader = Reader::from_path("src/data/data_as.csv").unwrap();
    let (imgs, datas) = load_initial_dataset(&mut reader);
    let data = imgs.into_iter().zip(datas.into_iter());

    let vs = nn::VarStore::new(Device::cuda_if_available());
    // let net = InitialIdentificationCnn::new(&vs.root());
}

fn load_initial_dataset(reader: &mut Reader<File>) -> InitialDataset {
    let mut data = Array2::default((0, reader.headers().unwrap().len()));
    for record in reader.records() {
        let record = record.expect("Failed to read record");
        let mut row = Vec::new();
        for field in record.iter() {
            row.push(field.to_owned());
        }
        data.append(
            Axis(0),
            Array2::from_shape_vec((1, row.len()), row).unwrap().view(),
        )
        .expect("Failed to append row to loaded dataset");
    }

    // Split into first column and the rest
    let (img, rest) = data.view().split_at(Axis(1), 1);

    // Convert the rest to an array of u8s
    let rest = rest.mapv(|x| x.parse::<u8>().unwrap());

    (img.to_owned(), rest.to_owned())
}

/// Simply flips a few images vertically to save time in drawing them. This should be run only once
/// after each time the images change.
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
        let img = image::open(format!("src/data/{i}.png")).unwrap();
        let img = img.to_luma_alpha8();
        let flipped = imageops::flip_vertical(&img);
        flipped.save(format!("src/data/r{i}.png")).unwrap();
    }
    for i in 1..=10 {
        let img = image::open(format!("src/data/g{i}.png")).unwrap();
        let img = img.to_luma_alpha8();
        let flipped = imageops::flip_vertical(&img);
        flipped.save(format!("src/data/l{i}.png")).unwrap();
    }
}

const IMAGE_PATHS: [&str; 20] = [
    "src/data/1.png",
    "src/data/2.png",
    "src/data/3.png",
    "src/data/4.png",
    "src/data/5.png",
    "src/data/6.png",
    "src/data/7.png",
    "src/data/8.png",
    "src/data/9.png",
    "src/data/10.png",
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

    // If pixel is sufficiently dark compared to the surrounding pixels, it's good.
    // Sufficiency is determined by the `BRIGHTNESS_THRESHOLD` environment variable.
    // If the environment variable is not set, it defaults to 20.
    if brightness
        < env::var("BRIGHTNESS_THRESHOLD")
            .unwrap_or("100".to_string())
            .parse::<u8>()
            .expect("BRIGHTNESS_THRESHOLD must be an 8-bit unsigned integer")
            .saturating_sub(surrounding_brightness)
    {
        return true;
    }

    // Otherwise, it's not good
    false
}
