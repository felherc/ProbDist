/**
 * Copyright 2018 Felipe Hernández
 * 
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
 * in compliance with the License. You may obtain a copy of the License at
 * 
 * http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software distributed under the 
 * License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
 * express or implied. See the License for the specific language governing permissions and
 * limitations under the License.
 */

package probDist.multiVar;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Hashtable;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.CholeskyDecomposition;
import org.apache.commons.math3.linear.DiagonalMatrix;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.stat.correlation.Covariance;

import probDist.ContProbDist;
import probDist.Normal;
import probDist.multiVar.tools.ContMultiSample;
import utilities.MatUtil;
import utilities.geom.PointID;
import utilities.stat.ContSeries;

/**
 * This class represents multivariate normal (Gaussian) probability distributions which are
 * parameterized by a vector of mean values and a covariance matrix
 * @author Felipe Hernández
 */
public class MultiVarNormal extends MultiContProbDist
{
	
	// --------------------------------------------------------------------------------------------
	// Constants
	// --------------------------------------------------------------------------------------------
	
	/**
	 * Distribution type identifier
	 */
	public final static String ID = "Multivariate normal";
	
	/**
	 * Default value for {@link #minVariance}
	 */
	public static final double DEF_MIN_VARIANCE = 1E-10;
	
	/**
	 * The minimum value for the determinant of a matrix to consider it invertible
	 */
	public static final double INVERTIBLE_THRESHOLD = 1E-200;
	
	/**
	 * Error message: mean and C size mismatch
	 */
	public static final String ERR_MEAN_C_SIZE_MISMATCH = "The mean and the covariance matrix "
															+ "have mismatching sizes";

	/**
	 * Error message: mean and variance size mismatch
	 */
	public static final String ERR_MEAN_VARIANCE_SIZE_MISMATCH = "The mean and variance vectors "
																	+ "do not have the same size";

	/**
	 * Error message: C not symmetrical
	 */
	public static final String ERR_C_NOT_SYMMETRICAL = "The covariance matrix is not symmetrical";

	/**
	 * Error message: C not square
	 */
	public static final String ERR_C_NOT_SQUARE = "The covariance matrix is not square";
	
	// --------------------------------------------------------------------------------------------
	// Attributes
	// --------------------------------------------------------------------------------------------
	
	/**
	 * The vector with the mean value of each dimension <i>k</i>
	 */
	private double[] mean;
	
	/**
	 * The <i>k</i> x <i>k</i> covariance matrix of the distribution, with the marginal variance
	 * values in the diagonal, and the covariance terms elsewhere. If <code>null</code>, the 
	 * covariance is represented by the {@link #variance} vector instead, which contain the 
	 * marginal variances of the dimensions and assumes independence between the dimensions 
	 * (covariance equal to zero).
	 */
	private double[][] C;
	
	/**
	 * The vector with the marginal variance of each dimension <i>k</i>, that is the coefficients 
	 * in the diagonal of the covariance matrix. Represents a distribution of independent variables 
	 * (covariance values assumed to be zero). Used for memory saving. When <code>null</code>, the 
	 * variance is represented by {@link #C} instead, with he additional covariance terms.
	 */
	private double[] variance;
	
	/**
	 * Minimum allowed value for the covariance matrix before it is considered zero for likelihood
	 * and sampling computation purposes. Also used as the minimum value to assume that a
	 * determinant is practically zero (the matrix is singular), and if a mean value for a
	 * dimension of the distribution is practically zero.
	 */
	private double minVariance;
	
	/**
	 * The indices of the dimensions that have non-zero variance and are not perfectly linearly
	 * dependent on the others and, therefore, can be used for likelihood and sampling computations
	 * using inverse covariance matrix methods. <code>Null</code> if the indices have not been
	 * determined yet.
	 */
	private ArrayList<Integer> active;
	
	/**
	 * The indices of the dimensions that have zero variance and, therefore, cannot be used for 
	 * likelihood and sampling computations using inverse covariance matrix methods. 
	 * <code>Null</code> if the indices have not been determined yet.
	 */
	private ArrayList<Integer> zeroVar;
	
	/**
	 * The indices of the dimensions that have a perfect linear dependency on other variables in 
	 * the distribution and, therefore, cannot be used for likelihood and sampling computations
	 * using inverse covariance matrix methods. <code>null</code> if the indices have not been
	 * determined yet.
	 */
	private ArrayList<Integer> linearlyDependent;
	
	/**
	 * Stores the regression coefficients to predict the variables of the distribution from the
	 * others assuming a linear combination. The key of the hash table is the index of the
	 * dependent variable, and the list contains tuples with the index of the explanatory variables
	 * and the corresponding coefficient. The -1 index in the tuples represents the constant 
	 * coefficient.
	 */
	private Hashtable<Integer, ArrayList<PointID>> regressionCoeff;
	
	/**
	 * Stores a probability distribution that represents a reduced-dimensionality version of the
	 * distribution's kernel. The kernel only contains the dimensions in {@link #active}, making
	 * the covariance matrix invertible so that it can be used for density and sampling 
	 * computations. Lazy computation: <code>null</code> if the kernel has not been computed or if
	 * all the variables in the distribution are active.
	 */
	private MultiVarNormal invertible;
	
	/**
	 * True if the method {@link #reduceToPositiveDefinite()} is currently running and 
	 * {@link #active}, {@link #zeroVar}, {@link #linearlyDependent}, and {@link #invertible} 
	 * should not be consulted.
	 */
	private boolean computingInvertible;
	
	// --------------------------------------------------------------------------------------------
	// Constructors
	// --------------------------------------------------------------------------------------------
	
	/**
	 * @param mean {@link #mean}
	 * @param C {@link #C}
	 */
	public MultiVarNormal(double[] mean, double[][] C)
	{
		if (!MatUtil.isSquare(C))
			throw new IllegalArgumentException(ERR_C_NOT_SQUARE);
		
		MatUtil.enforceSymmetry(C);
		
		if (mean.length != C.length)
			throw new IllegalArgumentException(ERR_MEAN_C_SIZE_MISMATCH + ": mean.length = " 
												+ mean.length + "; C.length = " + C.length);
		
		type				= MultiContProbDist.NORMAL;
		this.mean			= mean;
		this.C				= C;
		variance			= null;
		minVariance			= DEF_MIN_VARIANCE;
		
		active				= null;
		zeroVar				= null;
		linearlyDependent	= null;
		regressionCoeff		= new Hashtable<>();
	}
	
	/**
	 * @param mean {@link #mean}
	 * @param variance {@link #variance}
	 */
	public MultiVarNormal(double[] mean, double[] variance)
	{
		if (mean.length != variance.length)
			throw new IllegalArgumentException(ERR_MEAN_VARIANCE_SIZE_MISMATCH+ ": mean.length = " 
										+ mean.length + "; variance.length = " + variance.length);
		
		type				= MultiContProbDist.NORMAL;
		this.mean			= mean;
		C					= null;
		this.variance		= variance;
		minVariance			= DEF_MIN_VARIANCE;
		
		active				= null;
		zeroVar				= null;
		linearlyDependent	= null;
		regressionCoeff		= new Hashtable<>();
	}
	
	/**
	 * Creates a distribution from a list of samples
	 * @param samples	The list of samples
	 */
	public MultiVarNormal(ArrayList<ContMultiSample> samples)
	{
		int variables					= samples.get(0).getValues().size();
		double[][] data					= new double[samples.size()][variables];
		for (int s = 0; s < samples.size(); s++)
		{
			ArrayList<Double> values	= samples.get(s).getValues();
			for (int d = 0; d < variables; d++)
				data[s][d]				= values.get(d);
		}
		initValues(data, false);
	}
	
	/**
	 * Creates a distribution from a list of samples
	 * @param samples	A matrix with the values of a number of samples. The rows represent
	 * 					samples and the columns variables.
	 */
	public MultiVarNormal(double[][] samples)
	{
		initValues(samples, false);
	}
	
	/**
	 * Creates a distribution from a list of samples
	 * @param samples		The list of samples
	 * @param diagonalBW	True if the dependencies between variables should be ignored (the
	 * 						covariance matrix is diagonal). False otherwise.
	 */
	public MultiVarNormal(ArrayList<ContMultiSample> samples, boolean diagonalBW)
	{
		int variables					= samples.get(0).getValues().size();
		double[][] data					= new double[samples.size()][variables];
		for (int s = 0; s < samples.size(); s++)
		{
			ArrayList<Double> values	= samples.get(s).getValues();
			for (int d = 0; d < values.size(); d++)
				data[s][d]				= values.get(d);
		}
		initValues(data, diagonalBW);
	}
	
	/**
	 * Creates a distribution from a list of samples
	 * @param samples		A matrix with the values of a number of samples. The rows represent
	 * 						samples and the columns variables.
	 * @param diagonalBW	True if the dependencies between variables should be ignored (the
	 * 						covariance matrix is diagonal). False otherwise.
	 */
	public MultiVarNormal(double[][] samples, boolean diagonalBW)
	{
		initValues(samples, diagonalBW);
	}
	
	/**
	 * Initializes {@link #mean} and {@link #C}
	 * @param samples		A matrix with the values of a number of samples. The rows represent
	 * 						samples and the columns variables.
	 * @param diagonalBW	True if the dependencies between variables should be ignored (the
	 * 						covariance matrix is diagonal). False otherwise.
	 */
	private void initValues(double[][] samples, boolean diagonalBW)
	{
		type					= MultiContProbDist.NORMAL;
		
		// Determine mean
		int size				= samples[0].length;
		ArrayList<ContSeries> series = new ArrayList<>();
		for		(int d = 0; d < size; d++)
			series.add(new ContSeries(false));
		for		(int s = 0; s < samples.length; s++)
			for	(int d = 0; d < size; d++)
				series.get(d).addValue(samples[s][d]);
		mean					= new double[size];
		for		(int d = 0; d < size; d++)
			mean[d]				= series.get(d).getMean();
		
		// Determine the covariance matrix
		if (diagonalBW)
		{
			variance			= new double[size];
			for		(int d = 0; d < size; d++)
				variance[d]		= series.get(d).getVar();
			C					= null;
		}
		else
		{
			Covariance cov		= new Covariance(samples);
			C					= cov.getCovarianceMatrix().getData();
			variance			= null;
		}
		
		active					= null;
		zeroVar					= null;
		linearlyDependent		= null;
		regressionCoeff			= new Hashtable<>();
		minVariance				= DEF_MIN_VARIANCE;
	}
	
	// --------------------------------------------------------------------------------------------
	// Methods
	// --------------------------------------------------------------------------------------------

	@Override
	public String getTypeString() 
	{
		return ID;
	}

	@Override
	public int getDimensionality() 
	{
		return mean.length;
	}

	/**
	 * @return {@link #mean}
	 */
	public double[] getMean() 
	{
		return mean;
	}

	/**
	 * @param mean {@link #mean}
	 */
	public void setMean(double[] mean)
	{
		this.mean			= mean;
		if (linearlyDependent != null)
			if (linearlyDependent.size() > 0)
			{
				active				= null;
				zeroVar				= null;
				linearlyDependent	= null;
				regressionCoeff		= new Hashtable<>();
			}
	}

	/**
	 * @return {@link #C}
	 */
	public double[][] getC()
	{
		return C;
	}

	/**
	 * @param C {@link #C}
	 */
	public void setC(double[][] C)
	{
		if (!MatUtil.isSymmetric(C))
			throw new IllegalArgumentException(ERR_C_NOT_SYMMETRICAL);
		
		if (!MatUtil.isSquare(C))
			throw new IllegalArgumentException(ERR_C_NOT_SQUARE);
		
		if (C[0].length != mean.length)
			throw new IllegalArgumentException(ERR_MEAN_C_SIZE_MISMATCH);
		
		this.C				= C;
		variance			= null;
		
		active				= null;
		zeroVar				= null;
		linearlyDependent	= null;
		regressionCoeff		= new Hashtable<>();
	}
	
	@Override
	public double[][] getCovariance() 
	{
		return getC();
	}

	/**
	 * @return {@link #variance}
	 */
	public double[] getVariance()
	{
		return variance;
	}

	/**
	 * @param variance {@link #variance}
	 */
	public void setVariance(double[] variance)
	{
		if (mean.length != variance.length)
			throw new IllegalArgumentException(ERR_MEAN_VARIANCE_SIZE_MISMATCH+ ": mean.length = " 
										+ mean.length + "; variance.length = " + variance.length);
		
		this.variance	= variance;
		C				= null;
	}

	/**
	 * @return {@link #minVariance}
	 */
	public double getMinVariance()
	{
		return minVariance;
	}

	/**
	 * @param minVariance {@link #minVariance}
	 */
	public void setMinVariance(double minVariance) 
	{
		this.minVariance = minVariance;
	}

	/**
	 * Works as {@link #getpdf(double[])}
	 */
	public double getpdf(ArrayList<Double> x)
	{
		return Math.exp(getLogpdf(x));
	}
	
	/**
	 * Computes the value of the probability density function at the provided point. Returns 
	 * {@link java.lang.Double#NaN} if the distribution has not been correctly defined.
	 * @param x The point at which to evaluate the probability density
	 * @return The value of the probability density function at the provided point
	 */
	public double getpdf(double[] x)
	{
		return Math.exp(getLogpdf(x));
	}
	
	/**
	 * Works as {@link #getpdf(double[], boolean)}
	 */
	public double getpdf(ArrayList<Double> x, boolean allowMarginalization)
	{
		return Math.exp(getLogpdf(x, allowMarginalization));
	}
	
	/**
	 * Computes the value of the probability density function at the provided point. Returns 
	 * {@link java.lang.Double#NaN} if the distribution has not been correctly defined.
	 * @param x The point at which to evaluate the probability density
	 * @param allowMarginalization True if the distribution has zero-variance dimensions and needs
	 * to be marginalized. False otherwise: avoids performing the check.
	 * @return The value of the probability density function at the provided point
	 */
	public double getpdf(double[] x, boolean allowMarginalization)
	{
		return Math.exp(getLogpdf(x, allowMarginalization));
	}
	
	/**
	 * Works as {@link #getLogpdf(double[])}
	 */
	public double getLogpdf(ArrayList<Double> x)
	{
		return getLogpdf(x, true);
	}
	
	/**
	 * Computes the value of the natural logarithm of the probability density function at the 
	 * provided point. Returns {@link java.lang.Double#NaN} if the distribution has not been 
	 * correctly defined.
	 * @param x The point at which to evaluate the probability density
	 * @return The value of the natural logarithm of the probability density function at the 
	 * provided point
	 */
	public double getLogpdf(double[] x)
	{
		return getLogpdf(x, true);
	}
	
	/**
	 * Works as {@link #getLogpdf(double[], boolean)}
	 */
	public double getLogpdf(ArrayList<Double> x, boolean allowMarginalization)
	{
		double[] arrX	= new double[x.size()];
		for (int i = 0; i < x.size(); i++)
			arrX[i]		= x.get(i);
		return getLogpdf(arrX, allowMarginalization);
	}
	
	/**
	 * Computes the value of the natural logarithm of the probability density function at the 
	 * provided point. Returns {@link java.lang.Double#NaN} if the distribution has not been 
	 * correctly defined.
	 * @param x The point at which to evaluate the probability density
	 * @param allowMarginalization True if the distribution has zero-variance dimensions and needs
	 * to be marginalized. False otherwise: avoids performing the check.
	 * @return The value of the natural logarithm of the probability density function at the 
	 * provided point
	 */
	public double getLogpdf(double[] x, boolean allowMarginalization)
	{
		int k							= mean.length;
		if (k == 0)
			return Double.NaN;
		
		if (C == null)
			return getLogpdfDiagonal(x, allowMarginalization);
		
		// Verify that the covariance matrix is positive definite
		if (allowMarginalization && !isCovarianceInvertible())
		{
			ArrayList<Integer> toActivate 	= new ArrayList<>();
			ArrayList<Double> margX			= new ArrayList<>();
			for (int i = 0; i < k; i++)
			{
				double variance_i			= Math.abs(C[i][i]);
				if (variance_i < minVariance)
				{
					if (Math.abs(x[i] - mean[i]) > 2*Math.sqrt(minVariance))
						return Double.NEGATIVE_INFINITY;
				}
				else
				{
					toActivate.add(i);
					margX.add(x[i]);
				}
			}
			if (toActivate.size() < k)
			{
				// Compute pdf from marginalized distribution
				MultiVarNormal marginal		= marginalize(toActivate);
				return marginal == null ? Double.NaN : marginal.getLogpdf(margX, false);
			}
		}
		
		// Prepare variables
		RealVector xvec					= new ArrayRealVector(x);
		RealVector miuvec				= new ArrayRealVector(mean);
		RealMatrix Cmat					= new Array2DRowRealMatrix(C);
		CholeskyDecomposition CCh 		= new CholeskyDecomposition(Cmat);
		double detC						= CCh.getDeterminant();
		
		// Perform computations
		double normlzr					= Math.sqrt(Math.pow(2*Math.PI, k)*detC);
		RealMatrix Cinv					= CCh.getSolver().getInverse();
		RealVector dev					= xvec.subtract(miuvec);
		RealVector temp					= Cinv.preMultiply(dev);
		double exp						= -0.5*temp.dotProduct(dev);
		return exp - Math.log(normlzr);
	}
	
	/**
	 * Computes the value of the probability density function at the provided point in case the
	 * covariance matrix is diagonal (that is, if the dimensions are independent) as defined by
	 * {@link #variance}. Returns {@link java.lang.Double#NaN} if the distribution has not been 
	 * correctly defined.
	 * @param x The point at which to evaluate the probability density
	 * @param allowMarginalization True if the distribution has zero-variance dimensions and needs
	 * to be marginalized. False otherwise: avoids performing the check.
	 * @return The value of the probability density function at the provided point
	 */
	private double getLogpdfDiagonal(double[] x, boolean allowMarginalization)
	{
		int k							= mean.length;
		
		if (variance == null)
			return Double.NaN;
		if (variance.length == 0)
			return Double.NaN;
		
		// Verify if any dimensions have zero variance
		if (allowMarginalization && !isCovarianceInvertible())
		{
			ArrayList<Integer> toActivate 	= new ArrayList<>();
			ArrayList<Double> margX			= new ArrayList<>();
			for (int i = 0; i < k; i++)
			{
				if (variance[i] < minVariance)
				{
					if (Math.abs(x[i] - mean[i]) > 2*Math.sqrt(minVariance))
						return 0.0;
				}
				else
				{
					toActivate.add(i);
					margX.add(x[i]);
				}
			}
			if (toActivate.size() < k)
			{
				// Compute pdf from marginalized distribution
				MultiVarNormal marginal		= marginalize(toActivate);
				return marginal == null ? Double.NaN : marginal.getLogpdf(margX, false);
			}
		}
		
		// Compute log pdf
		double logSum					= 0.0;
		for (int i = 0; i < k; i++)
		{
			double stDev				= Math.sqrt(variance[i]);
			double pdf					= Normal.computepdf(mean[i], stDev, x[i]);
			logSum						+= Math.log(pdf);
		}
		return logSum;
	}
	
	/**
	 * @param x The point to compute the distance from
	 * @return The Mahalanobis distance between the provided point and the {@link #mean}
	 */
	public double getMahalanobisDistance(double[] x)
	{
		int k						= mean.length;
		if (k == 0)
			return Double.NaN;
		
		// Prepare invertible matrix
		if (active == null && !computingInvertible)
			reduceToPositiveDefinite();
		while (computingInvertible)
		{
			// Wait for other thread to finish computation
		}
		
		if (active.size() < k)
		{
			double[] margX			= new double[active.size()];
			for (int a = 0; a < active.size(); a++)
				margX[a]			= x[active.get(a)];
			return invertible.getMahalanobisDistance(margX);
		}
		
		// Prepare data
		double[] devs				= new double[k];
		for (int i = 0; i < k; k++)
			devs[i]					= x[i] - mean[i];
		RealMatrix covariance		= C == null	? new DiagonalMatrix(variance)
												: new Array2DRowRealMatrix(C);
		
		// Apply matrix transformation
		double[][] matrix			= covariance.getData();
		double[][] lower			= MatUtil.cholesky(matrix);
		
		// Compute Mahalanobis distance
		double[] y					= MatUtil.solveLower(lower, devs);
		double mahalanobis			= 0.0;
		for (int i = 0; i < y.length; i++)
			mahalanobis				+= y[i]*y[i];
		return Math.sqrt(mahalanobis);
	}
	
	/**
	 * Marginalizes the distribution of one of the dimensions. That is, creates a reduced 
	 * distribution of only one of the original dimensions.
	 * <p>Time complexity: <i>O(number of samples)</i>
	 * <br>Space complexity: <i>O(number of samples)</i>
	 * @param dimension The index of the dimension to marginalize
	 * @return The marginalized normal distribution
	 */
	public Normal marginalize(int dimension)
	{
		if (dimension >= mean.length)
			return null;
		
		double marginalMean		= mean[dimension];
		double marginalStDev	= Double.NaN;
		if (C != null)
			marginalStDev		= C[dimension][dimension];
		else if (variance != null)
			marginalStDev		= variance[dimension];
		return new Normal(marginalMean, marginalStDev);
	}
	
	/**
	 * Marginalizes the distribution of a set of variables. That is, creates a reduced distribution 
	 * with only a set of the original dimensions.
	 * @param toActivate A set containing the indices of the dimensions to marginalize
	 * @return The marginalized multivariate normal distribution. <code>null</code> if the 
	 * activation parameter contained no dimensions to marginalize.
	 */
	public MultiVarNormal marginalize(ArrayList<Integer> toActivate)
	{
		// Verify dimensions to marginalize
		if (toActivate.isEmpty())
			return null;
		for (Integer dimension : toActivate)
			if (dimension > mean.length || dimension < 0)
				toActivate.remove(dimension);
		
		// Marginalize
		double[] margMean		= new double[toActivate.size()];
		if (C != null)
		{
			int[] toAcArr		= new int[toActivate.size()];
			for (int i = 0; i < toActivate.size(); i++)
			{
				int index		= toActivate.get(i);
				margMean[i]		= mean[index];
				toAcArr[i]		= index;
			}
			RealMatrix jointC	= MatrixUtils.createRealMatrix(C);
			RealMatrix margC	= jointC.getSubMatrix(toAcArr, toAcArr);
			return new MultiVarNormal(margMean, margC.getData());
		}
		else
		{
			double[] margVariance = new double[toActivate.size()];
			for (int i = 0; i < toActivate.size(); i++)
			{
				int index		= toActivate.get(i);
				margMean[i]		= mean[index];
				margVariance[i]	= variance[index];
			}
			return new MultiVarNormal(margMean, margVariance);
		}
	}
	
	@Override
	public MultiContProbDist conditional(ArrayList<Double> x) 
	{
		// TODO Auto-generated method stub
		return null;
	}
	
	/**
	 * Generates a random vector from the multivariate normal distribution
	 * @return A random vector from the multivariate normal distribution
	 */
	public double[] sample()
	{
		if (C == null)
			return sampleDiagonal();
		
		int k							= mean.length;
		double[] result					= new double[k];
		
		// Prepare invertible matrix
		if (active == null && !computingInvertible)
			reduceToPositiveDefinite();
		while (computingInvertible)
		{
			// Wait for other thread to finish computation
		}
		
		// Generate random values
		if (active.size() == k)
		{			
			double[] random				= new double[k];
			for (int i = 0 ; i < k ; i++)
				random[i]				= Normal.sample(0, 1);
			double[][] A				= MatUtil.cholesky(C);
			double[] mult				= MatUtil.multiply(A, random);
			for(int i = 0 ; i < k ; i++)
				result[i]				= mean[i] + mult[i];
			return result;
		}
		else
		{
			if (zeroVar.size() == k)
			{
				for (int d = 0; d < k; d++)
					result[d]			= mean[d];
			}
			else
			{
				double[] margSample		= invertible.sample();
				for (int m = 0; m < active.size(); m++)
					result[active.get(m)] = margSample[m];
			}
		}
		
		// Add zero-variance variables
		if (zeroVar != null)
			for (Integer index : zeroVar)
				result[index]			= mean[index];
		
		// Add linearly-dependent variables
		if (linearlyDependent != null)
			for (Integer index2 : linearlyDependent)
			{
				ArrayList<PointID> reg	= regressionCoeff.get(index2);
				double value			= 0.0;
				for (int t = 0; t < reg.size(); t++)
				{
					PointID point		= reg.get(t);
					int index			= point.x;
					value				+= index == -1 ? point.y : result[index]*point.y;
				}
				result[index2]			= value;
			}
		return result;
	}
	
	/**
	 * Generates a random vector from the multivariate normal distribution in case the
	 * covariance matrix is diagonal (that is, if the dimensions are independent) as defined by
	 * {@link #variance}.
	 * @return A random vector from the multivariate normal distribution
	 * @throws IOException 
	 */
	private double[] sampleDiagonal()
	{
		int size					= mean.length;
		double[] result				= new double[size];
		
		// Verify if any dimensions have zero variance
		ArrayList<Integer> toActivate	= new ArrayList<>();
		for (int i = 0; i < size; i++)
			if (variance[i] >= minVariance)
				toActivate.add(i);
			else				
				result[i]			= mean[i];
		int marginalSize			= toActivate.size();
		
		if (marginalSize < mean.length)
		{
			if (toActivate.isEmpty())
				return mean;
			
			// Generate sample from marginalized distribution
			MultiVarNormal marginal = marginalize(toActivate);
			double[] marginalSample	= marginal.sample();
			int i = 0;
			for (Integer index : toActivate)
			{
				result[index]		= marginalSample[i];
				i++;
			}
			return result;
		}

		// Generate sample
		for (int i = 0; i < size; i++)
			result[i]				= Normal.sample(mean[i], Math.sqrt(variance[i]));
		
		return result;
	}
	
	@Override
	public double[][] sampleMultiple(int count) 
	{
		double[][] samples			= new double[count][mean.length];
		for (int s = 0; s < count; s++)
			samples[s]				= sample();
		return samples;
	}
	
	/**
	 * Determines if there are any zero-variance dimensions in the bandwidth matrix and if there 
	 * are any perfectly linearly dependent dimensions. In both cases, these dimensions need to be
	 * isolated by reducing the covariance matrix before it can be used for any computations
	 * involving its inversion (otherwise it cannot be inverted since it would not be positive
	 * definite) For the dependent dimensions, the regression coefficients to compute their values 
	 * given the other dimensions are calculated. The {@link #active} and 
	 * {@link #linearlyDependent} lists are filled and the regression coefficients are stored in 
	 * {@link #regressionCoeff}. Then {@link invertible} is constructed.
	 */
	private synchronized void reduceToPositiveDefinite()
	{
		computingInvertible				= true;
		int k							= mean.length;
		active							= new ArrayList<>(k);
		zeroVar							= new ArrayList<>(k);
		linearlyDependent				= new ArrayList<>(k);
		regressionCoeff					= new Hashtable<>(k);
		double[][] mat1					= null;
		double[][] chol1				= null;
		for (int d = 0; d < k; d++)
		{
			if (C == null)
			{
				if (variance[d] >= minVariance)
					active.add(d);
				else
					zeroVar.add(d);
				continue;	// Linear dependency is inexistent if C is diagonal
			}
			else if (C[d][d] < minVariance)
			{
				zeroVar.add(d);
				continue;
			}
				
			if (mat1 != null)
			{
				// Prepare extended matrix
				int size				= active.size();
				double[][] mat2			= new double[size + 1][size + 1];
				for		(int i = 0; i < size; i++)
					for	(int j = 0; j < size; j++)
						mat2[i][j]		= mat1[i][j];
				double[] v				= new double[size];
				for (int c = 0; c < active.size(); c++)
				{
					double val			= C[d][active.get(c)];
					v[c]				= val;
					mat2[size][c]		= val;
					mat2[c][size]		= val;
				}
				mat2[size][size]		= C[d][d];
				
				try
				{
					// Compute the determinant of the extended matrix
					double[][] chol2	= MatUtil.choleskyStepWise(mat2, chol1);
					double detL			= MatUtil.getDeterminantTriangular(chol2);
					double determinant	= detL*detL;
					
					// Determine if the extended matrix is invertible
					if (!Double.isNaN(determinant) && determinant >= INVERTIBLE_THRESHOLD)
					{
						active.add(d);
						mat1			= mat2;
						chol1			= chol2;
						continue;
					}
				} catch (Exception e) {}
				
				// Compute regression coefficients and mark as dependent
				linearlyDependent.add(d);
				double[] coeffs			= MatUtil.luEvaluate(chol1, v);
				
				// Compute constant vector
				double[] means			= new double[size];
				for (int c = 0; c < active.size(); c++)
					means[c]			= mean[active.get(c)];
				double combination		= MatUtil.dotProduct(coeffs, means);
				double constant			= mean[d] - combination;
				
				// Save coefficients
				ArrayList<PointID> reg	= new ArrayList<>();
				reg.add(new PointID(-1, constant));
				for (int c = 0; c < active.size(); c++)
					if (coeffs[c] != 0.0)
						reg.add(new PointID(active.get(c), coeffs[c]));
				regressionCoeff.put(d, reg);
			}
			else	// First dimension: cannot be dependent, needs to be added
			{
				double var				= C[d][d];
				double[][] m1			= {{			var		}};
				double[][] m2			= {{Math.sqrt(	var)	}};
				mat1					= m1;
				chol1					= m2;
				active.add(d);
			}
		}
		
		// Create marginalized distribution with invertible covariance matrix
		if (active.size() > 0 && active.size() < mean.length && C != null)
		{
			double[] margMean				= new double[active.size()];
			for (int a = 0; a < active.size(); a++)
				margMean[a]					= mean[active.get(a)];
			invertible						= new MultiVarNormal(margMean, mat1);
			invertible.markCovarianceAsInvertible();
		}
		else
			invertible						= null;
		
		computingInvertible					= false;
	}

	/**
	 * Marks the distribution as having an invertible {@link #C} (that is, it does not have 
	 * zero-variance dimensions or perfectly linearly dependent dimensions) so that this check does 
	 * not need to be performed further. However, marking {@link #C} as invertible when it is 
	 * really not might lead to errors when computing probability densities and generating random 
	 * samples.
	 */
	public void markCovarianceAsInvertible()
	{
		active					= new ArrayList<>();
		synchronized (active)
		{
			zeroVar				= new ArrayList<>();
			linearlyDependent	= new ArrayList<>();
			for (int d = 0; d < mean.length; d++)
				active.add(d);
		}
	}

	/**
	 * @return True if {@link #C} is invertible; false otherwise (because of zero-variance 
	 * dimensions or linearly dependent ones)
	 */
	public boolean isCovarianceInvertible()
	{
		// Prepare invertible matrix
		if (active == null && !computingInvertible)
			reduceToPositiveDefinite();
		while (computingInvertible)
		{
			// Wait for other thread to finish computation
		}
		
		if (C != null)
			return zeroVar.size() == 0 && linearlyDependent.size() == 0;
		else
		{
			for (int d = 0; d < mean.length; d++)
				if (variance[d] < minVariance)
					return false;
			return true;
		}
	}

	/**
	 * Generates a random vector from a multivariate normal distribution
	 * @param mean The vector with the mean value of each dimension <i>k</i>
	 * @param C The <i>k</i> x <i>k</i> covariance matrix of the distribution
	 * @return A random vector from a multivariate normal distribution
	 */
	public static double[] sample(double[] mean, double[][] C)
	{
		MultiVarNormal normal = new MultiVarNormal(mean, C);
		return normal.sample();
	}
	
	/**
	 * Generates a random vector from a multivariate normal distribution of independent variables
	 * @param mean		The vector with the mean value of each dimension <i>k</i>
	 * @param variance	The vector with the marginal variance of each dimension <i>k</i>. Covariance
	 * 					values are assumed to be zero
	 * @return A random vector from a multivariate normal distribution
	 */
	public static double[] sample(double[] mean, double[] variance)
	{
		MultiVarNormal normal = new MultiVarNormal(mean, variance);
		return normal.sample();
	}
	
	/**
	 * Randomly generates a new distribution
	 * @param mean				The mean values for each dimension/variable
	 * @param variance			The variance for each dimension/variable
	 * @param sampleCount		The number of samples to generate the distribution from
	 * @param freePercent		The percentage of dimensions/variables that are independent
	 * @param dependencePercent	A distribution with the percentage of dependent covariance when
	 * 							generating each variable (variables are generated one by one by
	 * 							correlating them with the existing ones)
	 * @param regressionSize	A distribution with the number of variables a new one is dependent
	 * 							on
	 * @param weightDist		A distribution with the variance weight assigned to each variable
	 * 							a new one is dependent on (variance contribution is proportional to
	 * 							the weight)
	 * @return A distribution created from the randomly generated samples
	 */
	public static MultiVarNormal randomGenerate(double[] mean, double[] variance, int sampleCount, 
			double freePercent, ContProbDist dependencePercent, ContProbDist regressionSize, 
			ContProbDist weightDist)
	{
		int k							= mean.length;
		if (k != variance.length)
			throw new IllegalArgumentException(ERR_MEAN_VARIANCE_SIZE_MISMATCH+ ": mean.length = " 
										+ mean.length + "; variance.length = " + variance.length);
		
		double[][] samples				= new double[sampleCount][k];
		
		// Create dependent variables
		int	freeCount					= (int)	Math.max(0.0, 	freePercent*k	);
			freeCount					= (int)	Math.min(k, 	freeCount		);
		for (int d = 0; d < k - freeCount; d++)
		{
			// Determine level of dependence and prepare selector
			ArrayList<Integer> selector	= new ArrayList<>(d);
			for (int i = 0; i < d; i++)
				if (variance[i] > 0.0)
					selector.add(i);
			Collections.shuffle(selector);
			double	depPerc				= 		Math.max(0.0, 	dependencePercent.sample()	);
					depPerc				= 		Math.min(1.0, 	depPerc						);
			int		regSize				= (int)	Math.max(1.0,	regressionSize.sample()		);
					regSize				= (int)	Math.min(selector.size(), regSize			);
			
			// Determine weights
			PointID[] weights			= new PointID[regSize];
			double weightSum			= 0.0;
			for (int i = 0; i < regSize; i++)
			{
				int index				= selector.get(i);
				double weight			= Math.max(0.0, weightDist.sample());
				weightSum				+= weight;
				weights[i]				= new PointID(index, weight);
			}
			
			// Determine coefficients
			PointID[] coeffs			= new PointID[regSize];
			double totalVar				= variance[d];
			double explainedVar			= regSize > 0 ? depPerc*totalVar : 0.0;
			double freeVar				= totalVar - explainedVar;
			double combination			= 0.0;
			for (int i = 0; i < regSize; i++)
			{
				PointID point			= weights[i];
				int index				= point.x;
				double var				= explainedVar*point.y/weightSum;
				double otherVar			= variance[index];
				double coeff			= otherVar == 0.0 ? 0.0 : Math.sqrt(var/otherVar);
				coeffs[i]				= new PointID(index, coeff);
				combination				+= coeff*mean[index];
			}
			
			// Compute samples
			double constant				= mean[d] - combination;
			for (int s = 0; s < sampleCount; s++)
			{
				double value			= 0.0;
				for (int i = 0; i < regSize; i++)
				{
					int index			= coeffs[i].x;
					double coeff		= coeffs[i].y;
					value				+= coeff*samples[s][index];
				}
				value					+= Normal.sample(constant, Math.sqrt(freeVar));
				samples[s][d]			= value;
			}
		}
		
		// Create independent variables
		for (int d = k - freeCount; d < k; d++)
			for (int s = 0; s < sampleCount; s++)
				samples[s][d]			= Normal.sample(mean[d], Math.sqrt(variance[d]));
		
		// Shuffle samples and create distribution
		ArrayList<double[]> dimArray	= new ArrayList<>(k);
		for (int d = 0; d < k; d++)
		{
			double[] dimension			= new double[sampleCount];
			for (int s = 0; s < sampleCount; s++)
				dimension[s]			= samples[s][d];
			dimArray.add(dimension);
		}
		Collections.shuffle(dimArray);
		for (int d = 0; d < k; d++)
		{
			double[] dimensions			= dimArray.get(d);
			for (int s = 0; s < sampleCount; s++)
				samples[s][d]			= dimensions[s];
		}
		return new MultiVarNormal(samples);
	}
	
}
