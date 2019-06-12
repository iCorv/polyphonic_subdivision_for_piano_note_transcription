import tensorflow as tf
import ps_inference as infer
import ps_model
import configurations.ps_hyper_parameters as php


def get_estimator(net, model_dir):
    hparams = php.get_hyper_parameters(net)
    estimator = tf.estimator.Estimator(
        model_fn=ps_model.conv_net_model_fn,
        model_dir=model_dir,
        params=hparams)
    return estimator


def export_saved_model(net, model_dir, export_dir_base, frames, bins):
    serving_input_receiver_fn = infer.get_serving_input_fn(frames, bins)
    estimator = get_estimator(net, model_dir)
    estimator.export_savedmodel(export_dir_base=export_dir_base, serving_input_receiver_fn=serving_input_receiver_fn,strip_default_attrs=True)#, checkpoint_path=model_dir+"/*")


def convert_model_to_tflite(saved_model_dir, export_dir):
    #saved_model_dir = "Users/Jaedicke/tensorflow/one_octave_resnet/model_ResNet_fold_1/"

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                           tf.lite.OpsSet.SELECT_TF_OPS]
    tflite_model = converter.convert()
    open(export_dir, "wb").write(tflite_model)


#export_saved_model(5, 199)

def main():
    model_dir = "Users/Jaedicke/tensorflow/polyphonic_subdivision_for_piano_note_transcription/model_multitask_resnet_fold_1"
    export_dir_base = "Users/Jaedicke/tensorflow/polyphonic_subdivision_for_piano_note_transcription/model_multitask_resnet_fold_1/saved_model"
    net = "ResNet_v1_RNN"
    #infer.build_predictor(net, model_dir)
    export_saved_model(net, model_dir, 2000, 199)


if __name__ == "__main__":
    main()

