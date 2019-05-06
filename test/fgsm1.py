models = KerasModelWrapper(model)
# Craft adversarial examples using Fast Gradient Sign Method (FGSM)
fgsm = FastGradientMethod(models, sess=sess)
fgsm_params = {'eps': 0.3}
adv_x_f = fgsm.generate(x, **fgsm_params)
# adv_x_f = tf.stop_gradient(adv_x_f)
X_test_adv, = batch_eval(sess, [x], [adv_x_f], [X_test_scaled])