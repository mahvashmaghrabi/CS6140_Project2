import pandas as pd
import sys
import numpy as np


def pca(data, normalize=True):
    # assign to A the data as a numpy matrix
    A = data.to_numpy()
    cols = len(A[0])
    rows = len(A)
    print(A)

    # assign to m the mean values of the columns of A
    m = A.mean(axis=0)
    print("m -> Mean")
    print(m)

    # assign to D the difference matrix A - m
    D = A - m
    print("D -> difference: A - mean")
    print(D)

    if normalize:
        sd = np.std(D, axis=0)
        print("Standard deviation")
        print(sd)
    else:
        sd = [1 for x in range(0, cols)]
        print(sd)

    # Assign all 1s to the standard deviation vector (1 for each column)

    # Divide each column by its standard deviation vector
    #    (hint: this can be done as a single operation)
    D = np.divide(D, sd)
    print(
        "Matrix after dividing by standard deviation of each column (Normalized data)"
    )
    print(D)

    # assign to U, S, V the result of running np.svd on D, with full_matrices=False
    print("Performing SVD")
    U, S, V = np.linalg.svd(D)
    print(U)
    print("S")
    print(S)

    print("Eigenvalues")
    S_square = np.square(S)
    eigenvalues = np.divide(S_square, (rows - 1))
    print(eigenvalues)

    # the eigenvalues of cov(A) are the squares of the singular values (S matrix)
    #   divided by the degrees of freedom (N-1). The values are sorted.
    print("Eigenvectors")
    print(V)

    projected_data = np.matmul(V, np.transpose(D))
    print("projected data")
    print(np.transpose(projected_data))
    # project the data onto the eigenvectors. Treat V as a transformation
    #   matrix and right-multiply it by D transpose. The eigenvectors of A
    #   are the rows of V. The eigenvectors match the order of the eigenvalues.

    # create a new data frame out of the projected data
    # return the means, standard deviations, eigenvalues, eigenvectors, and projected data
    return projected_data, eigenvalues


if __name__ == "__main__":
    filename = sys.argv[1]
    df = pd.read_csv(filename)
    # data = df.loc[:, "X1":"X3"]
    data = df.loc[:, "TotalSteps":"SedentaryMinutes"]
    pca(data)
