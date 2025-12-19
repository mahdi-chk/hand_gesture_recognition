import os
import cv2
import numpy as np
import tensorflow as tf


def load_and_preprocess_image(path, target_size=(224, 224)):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (target_size[1], target_size[0]))
    img_arr = img_resized.astype(np.float32) / 255.0
    return img_arr, img_resized


def find_last_conv_layer(model):
    # Search recursively through model layers and nested models
    conv_layers = []

    def _gather(layers_list):
        for layer in layers_list:
            # If it's a model/container, recurse
            if hasattr(layer, 'layers') and getattr(layer, 'layers'):
                _gather(layer.layers)
            else:
                if isinstance(layer, tf.keras.layers.Conv2D):
                    conv_layers.append(layer)

    try:
        _gather(model.layers)
    except Exception:
        # Fallback: inspect attributes that look like convs
        for name in dir(model):
            attr = getattr(model, name)
            if isinstance(attr, tf.keras.layers.Conv2D):
                conv_layers.append(attr)

    if not conv_layers:
        return None
    return conv_layers[-1]


def make_gradcam_heatmap(model, img_array, pred_index=None):
    # img_array: batched input (1, H, W, C)
    last_conv = find_last_conv_layer(model)

    # Two strategies: (A) use model.input (functional models), (B) manual forward for subclassed models
    try:
        if last_conv is None:
            raise RuntimeError('No Conv2D layer found in model to compute Grad-CAM')

        # Strategy A: functional/Sequential models with defined input
        grad_model = tf.keras.models.Model([model.input], [last_conv.output, model.output])

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]

        grads = tape.gradient(class_channel, conv_outputs)

        # Mean intensity of the gradient over each channel
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0]
        pooled_grads = pooled_grads

        # Weight the channels by corresponding gradients
        heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    except Exception:
        # Strategy B: subclassed model without defined .input; attempt manual forward
        # Attempt to detect GestureRecognitionCNN-like attributes
        if all(hasattr(model, attr) for attr in [
            'conv1_1','batch_norm1_1','conv1_2','batch_norm1_2','pool1',
            'conv2_1','batch_norm2_1','conv2_2','batch_norm2_2','pool2',
            'conv3_1','batch_norm3_1','conv3_2','batch_norm3_2','pool3',
            'conv4_1','batch_norm4_1','conv4_2','batch_norm4_2'
        ]):
            # Recompute the conv outputs inside the tape so gradients connect
            with tf.GradientTape() as tape:
                x = tf.convert_to_tensor(img_array)

                x = model.conv1_1(x)
                x = model.batch_norm1_1(x, training=False)
                x = model.conv1_2(x)
                x = model.batch_norm1_2(x, training=False)
                x = model.pool1(x)

                x = model.conv2_1(x)
                x = model.batch_norm2_1(x, training=False)
                x = model.conv2_2(x)
                x = model.batch_norm2_2(x, training=False)
                x = model.pool2(x)

                x = model.conv3_1(x)
                x = model.batch_norm3_1(x, training=False)
                x = model.conv3_2(x)
                x = model.batch_norm3_2(x, training=False)
                x = model.pool3(x)

                # Last conv block (stop before pooling)
                conv_out = model.conv4_1(x)
                conv_out = model.batch_norm4_1(conv_out, training=False)
                conv_out = model.conv4_2(conv_out)
                conv_out = model.batch_norm4_2(conv_out, training=False)

                # Continue forward pass through remaining classification head
                x2 = model.pool4(conv_out)
                x2 = model.dropout4(x2, training=False)
                x2 = model.gap(x2)
                x2 = model.dense1(x2)
                x2 = model.batch_norm_dense(x2, training=False)
                x2 = model.dropout_dense(x2, training=False)
                preds = model.dense2(x2)
                if pred_index is None:
                    pred_index = tf.argmax(preds[0])
                class_channel = preds[:, pred_index]

            grads = tape.gradient(class_channel, conv_out)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

            conv_outputs = conv_out[0]
            pooled_grads = pooled_grads
            heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
        else:
            # Could not handle this model type
            raise RuntimeError('Unable to compute Grad-CAM for this model type')

    # Relu and normalize
    heatmap = np.maximum(heatmap, 0)
    max_val = np.max(heatmap) if np.max(heatmap) != 0 else 1e-8
    heatmap /= max_val
    # If heatmap is a Tensor convert to numpy
    if hasattr(heatmap, 'numpy'):
        heatmap = heatmap.numpy()
    return heatmap


def apply_heatmap_on_image(orig_img, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    heatmap = cv2.resize(heatmap, (orig_img.shape[1], orig_img.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlayed = np.uint8(orig_img * (1 - alpha) + heatmap_color * alpha)
    return overlayed


def gradcam_for_image(model, image_path, out_path=None, target_size=(224, 224), top_k=3):
    img_arr, img_resized = load_and_preprocess_image(image_path, target_size=target_size)
    input_tensor = np.expand_dims(img_arr, axis=0)

    preds = model.predict(input_tensor)
    top_indices = preds[0].argsort()[-top_k:][::-1]
    top_scores = preds[0][top_indices]
    top = list(zip(top_indices.tolist(), top_scores.tolist()))

    heatmap = make_gradcam_heatmap(model, input_tensor, pred_index=int(top_indices[0]))
    overlay = apply_heatmap_on_image(img_resized, heatmap)

    if out_path:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        # Save RGB image as PNG
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        cv2.imwrite(out_path, overlay_bgr)

    return {
        'top_predictions': top,
        'heatmap': heatmap,
        'overlay': overlay,
    }
