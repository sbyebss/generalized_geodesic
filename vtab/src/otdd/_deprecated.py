



# def sort_matrix_rowscols(M, rows_sort, cols_sort):
#     """ Sort matrix M according to order in rows_sort and cols_sort"""
#     M_sort = np.zeros_like(M)
#     rows = range(M.shape[0])
#     cols = range(M.shape[1])
#     for i,u in enumerate(rows_sort):
#         for j,v in enumerate(cols_sort):
#             M_sort[i,j] = M[rows.index(u),cols.index(v)]
#     return M_sort

        # ax.tick_params(
        #     axis='both',          # changes apply to the x-axis
        #     which='both',      # both major and minor ticks are affected
        #     bottom=False,      # ticks along the bottom edge are off
        #     top=False,         # ticks along the top edge are off
        #     labelbottom=False) # labels along the bottom edge are off

    # def plot_label_similarity(self, plot_means=False):
    #     Means, Covs = self._get_label_stats()
    #     fig, ax = plt.subplots(1,10,figsize=(12,1))
    #     for i, m in enumerate(Means):
    #         ax[i].imshow(m.reshape(self.input_size), cmap='Greys')
    #         ax[i].axis('off')
    #     plt.show()
