pub mod sys;
mod tflite;

pub use self::tflite::*;

type Type = sys::TfLiteType;
type XnnPackDelegateOptions = sys::TfLiteXNNPackDelegateOptions;

#[derive(Debug)]
pub enum Error {
    FailedToLoad,
    Generic,
    ErrorStatus(sys::TfLiteStatus),
}
