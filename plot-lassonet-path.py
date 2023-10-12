import pickle

import numpy as np
import matplotlib.pyplot as plt


with open('lassonet-path.pickle', 'rb') as f:
	data = pickle.load(f)

path = np.asarray([save.selected.cpu().data.numpy() for save in data])
loss = np.asarray([save.val_loss for save in data])
lambda_ = np.asarray([save.lambda_ for save in data])
lengths = np.sum(path, axis=0)

print(np.sum(path[:, :6], axis=0))

plt.figure(figsize=(10, 7))

ax = plt.subplot(2, 1, 1)
plt.plot(np.sum(path, axis=1), color='black')

for k in [13, 48, 77, 94, 158]:
	print(path[10, :6])
	plt.axvline(x=k, color='mediumseagreen', linestyle='--')
	text = ' ' + str(list(np.where(path[k, :6])[0]))
	plt.annotate(text, (k, 200))
plt.axvline(x=np.argmin(loss), color='orangered', linestyle='--')
text = ' ' + str(list(np.where(path[np.argmin(loss), :6])[0]))
plt.annotate(text, (np.argmin(loss), 200))


ax.spines[['right', 'top']].set_visible(False)
plt.xlabel('Iterations')
plt.ylabel('Number of active features')
plt.grid(alpha=0.4, linestyle='--', linewidth=0.5, color='grey')

ax = plt.subplot(2, 1, 2)

plt.plot(loss, color='black')
plt.axvline(x=np.argmin(loss), color='orangered', linestyle='--')

ax.spines[['right', 'top']].set_visible(False)
plt.xlabel('Iterations')
plt.ylabel('Validation loss')
plt.grid(alpha=0.4, linestyle='--', linewidth=0.5, color='grey')

plt.tight_layout()

# plt.imshow(path)
plt.savefig('lassonet-path.png', dpi=300)
# plt.show()

print(path)
