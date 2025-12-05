
# # ------------------------------------------------------
# # 1. Load dataset
# # ------------------------------------------------------
# train_data = USPS(root='./USPS/', train=True, download=True)
# test_data  = USPS(root='./USPS/', train=False, download=True)

# X_train = train_data.data        # already numpy: shape (N, 16, 16)
# y_train = torch.tensor(train_data.targets, dtype=torch.long)

# # ------------------------------------------------------
# # 2. Flatten images to (N, 256)
# # ------------------------------------------------------
# X_train_np = X_train.reshape(len(X_train), -1)  # remove .numpy()

# # ------------------------------------------------------
# # 3. Apply VAT
# # ------------------------------------------------------
# D_reordered, vat_indices = VAT(X_train_np)

# # ------------------------------------------------------
# # 4. Reorder dataset
# # ------------------------------------------------------
# X_train_vat = X_train_np[vat_indices]
# y_train_vat = y_train[vat_indices]

# print("Original:", X_train_np.shape)
# print("Reordered:", X_train_vat.shape)
# print("VAT indices sample:", vat_indices[:10])