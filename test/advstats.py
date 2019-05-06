print("====================== Adversarial Feature Statistics =======================")

feats = dict()
total = 0
orig_attack = X_test_scaled - X_adv
for i in range(0, orig_attack.shape[0]):
	ind = np.where(orig_attack[i, :] != 0)[0]
	total += len(ind)
	for j in ind:
		if j in feats:
			feats[j] += 1
		else:
			feats[j] = 1

# The number of features that were changed for the adversarial samples
print("Number of unique features changed with JSMA: {}".format(len(feats.keys())))
print("Number of average features changed per datapoint with JSMA: {}".format(total/len(orig_attack)))

top_10 = sorted(feats, key=feats.get, reverse=True)[:10]
top_20 = sorted(feats, key=feats.get, reverse=True)[:20]
print("Top ten features: ", X_test.columns[top_10])

top_10_val = [100*feats[k] / y_test.shape[0] for k in top_10]
top_20_val = [100*feats[k] / y_test.shape[0] for k in top_20]

plt.figure(figsize=(12, 6))
plt.bar(np.arange(20), top_20_val, align='center')
plt.xticks(np.arange(20), X_test.columns[top_20], rotation='vertical')
plt.title('Feature participation in adversarial examples')
plt.ylabel('Percentage (%)')
plt.xlabel('Features')
plt.savefig('Adv_features.png', bbox_inches = "tight")