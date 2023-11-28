import matplotlib.pyplot as plt

def bar_plot_template():
	fig, ax = plt.subplots(figsize=(18, 7))

	ax.legend()
	ax.set_ylabel('Change in \%')
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.spines['bottom'].set_visible(False)
	ax.spines['left'].set_visible(False)
	ax.grid(axis='y')
	ax.set_ylim(-50, 80)

	return ax

def savefig(*args, **kwargs):
	kwargs['bbox_inches'] = 'tight'
	plt.savefig(*args, **kwargs)
