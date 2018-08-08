namespace Accord.Statistics.Analysis
{
    using System;
    using System.Collections.ObjectModel;
    using Accord.Math;
    using Accord.Math.Comparers;
    using Accord.Math.Decompositions;
    using Accord.MachineLearning;
    using Accord.Statistics.Analysis.Base;
    using Accord.Statistics.Models.Regression.Linear;
    using Accord.Compat;

   
    [Serializable]
#pragma warning disable 612, 618
    public class PrincipalComponentAnalysis : BasePrincipalComponentAnalysis, ITransform<double[], double[]>,
        IUnsupervisedLearning<MultivariateLinearRegression, double[], double[]>,
        IMultivariateAnalysis, IProjectionAnalysis
#pragma warning restore 612, 618
    {

       
        [Obsolete("Please pass the 'data' matrix to the Learn method instead.")]
        public PrincipalComponentAnalysis(double[][] data, AnalysisMethod method = AnalysisMethod.Center)
        {
            if (data == null)
                throw new ArgumentNullException("data");

            if (data.Length == 0)
                throw new ArgumentException("Data matrix cannot be empty.", "data");

            int cols = data[0].Length;
            for (int i = 0; i < data.Length; i++)
            {
                if (data[i].Length != cols)
                    throw new DimensionMismatchException("data",
                        "Matrix must be rectangular. The vector at position " + i +
                        " has a different length than other vectors");
            }

            this.array = data;
            this.Method = (PrincipalComponentMethod)method;
            this.NumberOfInputs = cols;
            this.NumberOfOutputs = data.Columns();
        }

        
        [Obsolete("Please pass the 'data' matrix to the Learn method instead.")]
        public PrincipalComponentAnalysis(double[,] data, AnalysisMethod method = AnalysisMethod.Center)
        {
            if (data == null)
                throw new ArgumentNullException("data");

            this.source = data;
            this.Method = (PrincipalComponentMethod)method;
            this.NumberOfInputs = data.Rows();
            this.NumberOfOutputs = data.Columns();
        }

         PrincipalComponentAnalysis(PrincipalComponentMethod method = PrincipalComponentMethod.Center,
            bool whiten = false, int numberOfOutputs = 0)
        {
            this.Method = method;
            this.Whiten = whiten;
            this.NumberOfOutputs = numberOfOutputs;
        }


       
        public MultivariateLinearRegression Learn(double[][] x, double[] weights = null)
        {
            this.NumberOfInputs = x.Columns();

            if (Method == PrincipalComponentMethod.Center || Method == PrincipalComponentMethod.Standardize)
            {
                if (weights == null)
                {
                    this.Means = x.Mean(dimension: 0);

                    double[][] matrix = Overwrite ? x : Jagged.CreateAs(x);
                    x.Subtract(Means, dimension: (VectorType)0, result: matrix);

                    if (Method == PrincipalComponentMethod.Standardize)
                    {
                        this.StandardDeviations = x.StandardDeviation(Means);
                        matrix.Divide(StandardDeviations, dimension: (VectorType)0, result: matrix);
                    }

                    
                    var svd = new JaggedSingularValueDecomposition(matrix,
                        computeLeftSingularVectors: false,
                        computeRightSingularVectors: true,
                        autoTranspose: true, inPlace: true);

                    SingularValues = svd.Diagonal;
                    Eigenvalues = SingularValues.Pow(2);
                    Eigenvalues.Divide(x.Rows() - 1, result: Eigenvalues);
                    ComponentVectors = svd.RightSingularVectors.Transpose();
                }
                else
                {
                    this.Means = x.WeightedMean(weights: weights);

                    double[][] matrix = Overwrite ? x : Jagged.CreateAs(x);
                    x.Subtract(Means, dimension: (VectorType)0, result: matrix);

                    if (Method == PrincipalComponentMethod.Standardize)
                    {
                        this.StandardDeviations = x.WeightedStandardDeviation(weights, Means);
                        matrix.Divide(StandardDeviations, dimension: (VectorType)0, result: matrix);
                    }

                    double[,] cov = x.WeightedCovariance(weights, Means);

                    
                    var evd = new EigenvalueDecomposition(cov,
                        assumeSymmetric: true, sort: true);

                    
                    Eigenvalues = evd.RealEigenvalues;
                    SingularValues = Eigenvalues.Sqrt();
                    ComponentVectors = Jagged.Transpose(evd.Eigenvectors);
                }
            }
            else if (Method == PrincipalComponentMethod.CovarianceMatrix
                  || Method == PrincipalComponentMethod.CorrelationMatrix)
            {
                if (weights != null)
                    throw new Exception();

                var evd = new JaggedEigenvalueDecomposition(x,
                    assumeSymmetric: true, sort: true);

                Eigenvalues = evd.RealEigenvalues;
                SingularValues = Eigenvalues.Sqrt();
                ComponentVectors = evd.Eigenvectors.Transpose();
            }
            else
            {
                throw new InvalidOperationException("Invalid method, this should never happen: {0}".Format(Method));
            }

            if (Whiten)
                ComponentVectors.Divide(SingularValues, dimension: (VectorType)1, result: ComponentVectors);

            CreateComponents();

            return CreateRegression();
        }

        private MultivariateLinearRegression CreateRegression()
        {
            double[][] weights = ComponentVectors;
            if (Method == PrincipalComponentMethod.Standardize || Method == PrincipalComponentMethod.CorrelationMatrix)
                weights = weights.Divide(StandardDeviations, dimension: (VectorType)0);

            double[] bias = weights.Dot(Means).Multiply(-1);

            return new MultivariateLinearRegression()
            {
                Weights = weights.Transpose(),
                Intercepts = bias
            };
        }

        [Obsolete("Please use the Learn method instead.")]
        public virtual void Compute()
        {
            if (!onlyCovarianceMatrixAvailable)
            {
                int rows;

                if (this.array != null)
                {
                    rows = array.Length;

                    double[][] matrix = Adjust(array, Overwrite);

                    var svd = new JaggedSingularValueDecomposition(matrix,
                        computeLeftSingularVectors: true,
                        computeRightSingularVectors: true,
                        autoTranspose: true,
                        inPlace: true);

                    SingularValues = svd.Diagonal;

                    ComponentVectors = svd.RightSingularVectors.Transpose();
                }
                else
                {
                    rows = source.GetLength(0);

#pragma warning disable 612, 618
                    double[,] matrix = Adjust(source, Overwrite);
#pragma warning restore 612, 618

                    var svd = new SingularValueDecomposition(matrix,
                        computeLeftSingularVectors: true,
                        computeRightSingularVectors: true,
                        autoTranspose: true,
                        inPlace: true);

                    SingularValues = svd.Diagonal;
                    ComponentVectors = svd.RightSingularVectors.ToArray().Transpose();

                }

                Eigenvalues = new double[SingularValues.Length];
                for (int i = 0; i < SingularValues.Length; i++)
                    Eigenvalues[i] = SingularValues[i] * SingularValues[i] / (rows - 1);
            }
            else
            {
                var evd = new EigenvalueDecomposition(covarianceMatrix,
                    assumeSymmetric: true,
                    sort: true);

                Eigenvalues = evd.RealEigenvalues;
                var eigenvectors = evd.Eigenvectors.ToJagged();
                SingularValues = Eigenvalues.Sqrt();
                ComponentVectors = eigenvectors.Transpose();
            }

            if (Whiten)
            {
                ComponentVectors = ComponentVectors.Transpose().Divide(Eigenvalues, dimension: 0).Transpose();
            }

            CreateComponents();

            if (!onlyCovarianceMatrixAvailable)
            {
                if (array != null)
                    result = Transform(array).ToMatrix();
                else if (source != null)
                    result = Transform(source.ToJagged()).ToMatrix();
            }
        }

        public override double[][] Transform(double[][] data, double[][] result)
        {
            if (ComponentVectors == null)
                throw new InvalidOperationException("The analysis must have been computed first.");

            int rows = data.Rows();
#pragma warning disable 612, 618
            data = Adjust(data, false);
#pragma warning restore 612, 618
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < NumberOfOutputs; j++)
                    for (int k = 0; k < ComponentVectors[j].Length; k++)
                        result[i][j] += data[i][k] * ComponentVectors[j][k];

            return result;
        }

       
        [Obsolete("Please use Jagged matrices instead.")]
        public virtual double[,] Revert(double[,] data)
        {
            return Revert(data.ToJagged()).ToMatrix();
        }

        public virtual double[][] Revert(double[][] data)
        {
            if (data == null)
                throw new ArgumentNullException("data");

            int rows = data.Rows();
            int cols = data.Columns();
            int components = NumberOfOutputs;
            double[][] reversion = Jagged.Zeros(rows, components);

            // Revert the data (reversion = data * eigenVectors.Transpose())
            for (int i = 0; i < components; i++)
                for (int j = 0; j < rows; j++)
                    for (int k = 0; k < cols; k++)
                        reversion[j][i] += data[j][k] * ComponentVectors[k][i];


            if (this.Method == PrincipalComponentMethod.Standardize || this.Method == PrincipalComponentMethod.CorrelationMatrix)
                reversion.Multiply(StandardDeviations, dimension: (VectorType)0, result: reversion);

            reversion.Add(Means, dimension: (VectorType)0, result: reversion);
            return reversion;
        }



        
        [Obsolete("This method is obsolete.")]
        protected internal double[,] Adjust(double[,] matrix, bool inPlace)
        {
            if (Means == null || Means.Length == 0)
            {
                Means = matrix.Mean(dimension: 0);
                StandardDeviations = matrix.StandardDeviation(Means);
            }

            
            double[,] result = matrix.Center(Means, inPlace);

            if (this.Method == PrincipalComponentMethod.Standardize || this.Method == PrincipalComponentMethod.CorrelationMatrix)
            {
                result.Standardize(StandardDeviations, true);
            }

            return result;
        }

        [Obsolete("This method is obsolete.")]
        protected internal double[][] Adjust(double[][] matrix, bool inPlace)
        {
            if (Means == null || Means.Length == 0)
            {
                Means = matrix.Mean(dimension: 0);
                StandardDeviations = matrix.StandardDeviation(Means);
            }
            double[][] result = matrix.Center(Means, inPlace);

            if (this.Method == PrincipalComponentMethod.Standardize || this.Method == PrincipalComponentMethod.CorrelationMatrix)
            {
                result.Standardize(StandardDeviations, true);
            }

            return result;
        }





        #region Named Constructors
        
        public static PrincipalComponentAnalysis FromCovarianceMatrix(double[] mean, double[,] covariance)
        {
            if (mean == null)
                throw new ArgumentNullException("mean");
            if (covariance == null)
                throw new ArgumentNullException("covariance");

            if (covariance.GetLength(0) != covariance.GetLength(1))
                throw new NonSymmetricMatrixException("Covariance matrix must be symmetric");

            var pca = new PrincipalComponentAnalysis(method: PrincipalComponentMethod.CovarianceMatrix);
            pca.Means = mean;
            pca.covarianceMatrix = covariance;
            pca.onlyCovarianceMatrixAvailable = true;
            pca.NumberOfInputs = covariance.GetLength(0);
            pca.NumberOfOutputs = covariance.GetLength(0);
            return pca;
        }



        public static PrincipalComponentAnalysis FromCorrelationMatrix(double[] mean, double[] stdDev, double[,] correlation)
        {
            if (!correlation.IsSquare())
                throw new NonSymmetricMatrixException("Correlation matrix must be symmetric");

            var pca = new PrincipalComponentAnalysis(method: PrincipalComponentMethod.CorrelationMatrix);
            pca.Means = mean;
            pca.StandardDeviations = stdDev;
            pca.covarianceMatrix = correlation;
            pca.onlyCovarianceMatrixAvailable = true;
            pca.NumberOfInputs = correlation.GetLength(0);
            pca.NumberOfOutputs = correlation.GetLength(0);
            return pca;
        }

    
        public static PrincipalComponentAnalysis FromGramMatrix(double[] mean, double[] stdDev, double[,] kernelMatrix)
        {
            if (!kernelMatrix.IsSquare())
                throw new NonSymmetricMatrixException("Correlation matrix must be symmetric");

            var pca = new PrincipalComponentAnalysis(method: PrincipalComponentMethod.KernelMatrix);
            pca.Means = mean;
            pca.StandardDeviations = stdDev;
            pca.covarianceMatrix = kernelMatrix;
            pca.onlyCovarianceMatrixAvailable = true;
            pca.NumberOfInputs = mean.GetLength(0);
            return pca;
        }

       
        public static double[][] Reduce(double[][] x, int dimensions)
        {
            return new PrincipalComponentAnalysis()
            {
                NumberOfOutputs = dimensions
            }.Learn(x).Transform(x);
        }
        #endregion

    }


  
    [Serializable]
    public class PrincipalComponent : IAnalysisComponent
    {

        private int index;
        private BasePrincipalComponentAnalysis principalComponentAnalysis;


        internal PrincipalComponent(BasePrincipalComponentAnalysis analysis, int index)
        {
            this.index = index;
            this.principalComponentAnalysis = analysis;
        }


     
        public int Index
        {
            get { return this.index; }
        }

       
        public BasePrincipalComponentAnalysis Analysis
        {
            get { return this.principalComponentAnalysis; }
        }

      
        public double Proportion
        {
            get { return this.principalComponentAnalysis.ComponentProportions[index]; }
        }

       
        public double CumulativeProportion
        {
            get { return this.principalComponentAnalysis.CumulativeProportions[index]; }
        }

        
        public double SingularValue
        {
            get { return this.principalComponentAnalysis.SingularValues[index]; }
        }

      
        public double Eigenvalue
        {
            get { return this.principalComponentAnalysis.Eigenvalues[index]; }
        }

        public double[] Eigenvector
        {
            get { return principalComponentAnalysis.ComponentVectors[index]; }
        }
    }

   
    [Serializable]
    public class PrincipalComponentCollection : ReadOnlyCollection<PrincipalComponent>
    {
        internal PrincipalComponentCollection(PrincipalComponent[] components)
            : base(components)
        {
        }
    }

}
