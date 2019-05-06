models = KerasModelWrapper(model)
jsma = SaliencyMapMethod(models, sess=sess)
jsma_params = {'theta': 1., 'gamma': 0.1, 'clip_min': 0., 'clip_max': 1., 'y_target': None}

for sample_ind in range(0, source_samples):
	sample = X_test_scaled[sample_ind: (sample_ind+1)]
	# We want to find an adversarial example for each possible target class
	# (i.e. all classes that differ from the label given in the dataset)
	current_class = int(np.argmax(y_test[sample_ind]))

	# Only target the normal class
	for target in [0]:
		if current_class == 0:
			break

		print('Generating adv. example for target class {} for sample {}'.format(target, sample_ind), end='\r')

		# Run the Jacobian-based saliency map approach
		one_hot_target = np.zeros((1, FLAGS.nb_classes), dtype=np.float32)
		one_hot_target[0, target] = 1
		jsma_params['y_target'] = one_hot_target
		adv_x = jsma.generate_np(sample, **jsma_params)

		# Check if success was achieved
		res = int(model_argmax(sess, x, predictions, adv_x) == target)

		# Compute number of modified features
		adv_x_reshape = adv_x.reshape(-1)
		test_in_reshape = X_test_scaled[sample_ind].reshape(-1)
		nb_changed = np.where(adv_x_reshape != test_in_reshape)[0].shape[0]
		percent_perturb = float(nb_changed) / adv_x.reshape(-1).shape[0]

		X_adv[sample_ind] = adv_x
		results[target, sample_ind] = res
		perturbations[target, sample_ind] = percent_perturb

print()
print(X_adv.shape)