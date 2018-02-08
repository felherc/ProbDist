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
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import probDist.KernelDensity;
import probDist.Normal;
import probDist.multiVar.tools.ContMultiSample;
import probDist.multiVar.tools.Sample;
import utilities.MatUtil;
import utilities.Utilities;
import utilities.geom.PointID;
import utilities.stat.ContPairedSeries;
import utilities.stat.ContStats;

/**
 * This class represents a multi-variable continuous probability distribution which is modeled
 * using kernel density estimation over a set of samples. S-class, D-class, and F-class Gaussian
 * kernels can be used.
 * @author Felipe Hernández
 */
public class MultiVarKernelDensity extends NonParametric
{

	// --------------------------------------------------------------------------------------------
	// Constants
	// --------------------------------------------------------------------------------------------
	
	/**
	 * Distribution type identifier
	 */
	public final static String ID = "Kernel density";
	
	/**
	 * Default value for {@link #weighted}
	 */
	public final static boolean DEF_WEIGHTED = false;
	
	/**
	 * Default value for {@link #minVariance}
	 */
	public static final double DEF_MIN_VARIANCE = 1E-10;
	
	/**
	 * The minimum value for the determinant of a matrix to consider it invertible
	 */
	public static final double INVERTIBLE_THRESHOLD = 1E-200;
	
	/**
	 * Percentage of the standard deviation of a linearly dependent variable to use as a threshold
	 * to accept a given point as potentially likely given the distribution 
	 */
	public static final double DEPENDENCY_ACCEPTANCE = 2.0;

	/**
	 * Error message: Bandwidth undefined
	 */
	public final static String ERR_BW_NOT_DEFINED = "No bandwidth has been defined for the "
														+ "distribution";

	/**
	 * Error message: Bandwidth diagonal vector is too small
	 */
	public final static String ERR_BW_VECTOR_SMALL = "The diagonal should be of size ";
	
	/**
	 * Error message: Matrix is too small
	 */
	public final static String ERR_BW_MATRIX_SMALL = "The matrix should be of size ";
	
	/**
	 * Error message: Matrix is not symmetrical
	 */
	public final static String ERR_NOT_SYMMETRIC = "The matrix is not symmetrical";

	/**
	 * Error message: No samples in the distribution
	 */
	public static final String ERR_NO_SAMPLES = "No samples have been defined";
	
	/**
	 * Error message: Particle of wrong size
	 */
	public static final String ERR_PARTICLE_SIZE = "The particle does not have the same number "
								+ "of dimensions (%1$d) as the samples in the distribution (%2$d)";

	/**
	 * Error message: Bandwidth matrix too big; cannot create instance
	 */
	public static final String ERR_MATRIX_TOO_BIG = "The covariance matrix is too big and cannot "
														+ "be instantiated";

	/**
	 * Error message: an offered sample contains a {@link java.lang.Double#NaN} value
	 */
	public static final String ERR_VALUE_IS_NAN = "The value on index %1$d of the offered sample "
													+ "is not a number (NaN)";
	
	// --------------------------------------------------------------------------------------------
	// Attributes
	// --------------------------------------------------------------------------------------------
	
	/**
	 * True if the samples have different weights. False if every sample has the same weight. That 
	 * is, if each sample has a kernel with the same contribution to the probability density.
	 */
	private boolean weighted;
	
	/**
	 * List of the samples to estimate the distribution from
	 */
	private ArrayList<ContMultiSample> samples;
	
	/**
	 * List with the values in the samples organized for each dimension. Used for computing the
	 * means, standard deviations, and other statistics from the data.
	 */
	private ArrayList<ContStats> statistics;
	
	/**
	 * Scalar value to define a global bandwidth for all dimensions (<i>S</i>-class kernels): 
	 * all dimensions are assumed independent in the kernel with equal variance (as if using
	 * the identity matrix multiplied by this scalar as the bandwidth matrix).
	 * <code>Double.NaN</code> if <i>S</i>-class kernels should not be used (bandwidth defined
	 * either by {@link #diagonalBandwidth} or {@link #invertible} instead).
	 */
	private double scalarBandwidth;
	
	/**
	 * Vector of values with the bandwidths for every dimension (<i>D</i>-class kernels): all
	 * dimensions are assumed independent in the kernel (as if using a diagonal matrix as the
	 * bandwidth matrix). <code>null</code> if <i>D</i>-class kernels should not be used (bandwidth 
	 * defined either by {@link #scalarBandwidth} or {@link #invertible} instead).
	 */
	private double[] diagBandwidth;
	
	/**
	 * List of dimension indices that have non-zero variance. Lazy computation: can be 
	 * <code>null</code> if it has not been computed.
	 */
	private ArrayList<Integer> active;
	
	/**
	 * The indices of the dimensions that have zero variance and, therefore, cannot be used for 
	 * likelihood and sampling computations using inverse covariance matrix methods. Lazy 
	 * computation: <code>null</code> if the indices have not been determined yet.
	 */
	private ArrayList<Integer> zeroVar;
	
	/**
	 * The indices of the dimensions that have a perfect linear dependency on other variables in 
	 * the distribution and, therefore, cannot be used for likelihood and sampling computations
	 * using inverse covariance matrix methods. Lazy computation: <code>null</code> if the indices 
	 * have not been determined yet.
	 */
	private ArrayList<Integer> linearlyDependent;
	
	/**
	 * Stores the regression coefficients to predict the variables of the distribution from the
	 * others assuming a linear combination. The key of the hash table is the index of the
	 * dependent variable, and the list contains tuples with the index of the explanatory variables
	 * and the corresponding coefficient. The -1 index in the tuples represents the constant 
	 * coefficient. The -2 index represents the independent portion of the covariance for when the 
	 * size of {@link #invertible} is limited (assume zero if not present).
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
	 * List of marginalized probability distributions. The list corresponds to the dimensions in
	 * {@link #active}. Lazy computation: <code>null</code> if the marginals have not been 
	 * computed.
	 */
	private ArrayList<KernelDensity> marginals;
	
	/**
	 * Minimum allowed value for the covariance matrix before it is considered zero for likelihood
	 * and sampling computation purposes
	 */
	private double minVariance;
	
	// --------------------------------------------------------------------------------------------
	// Constructors
	// --------------------------------------------------------------------------------------------
	
	public MultiVarKernelDensity()
	{
		initialize(DEF_WEIGHTED);
	}
	
	/**
	 * @param weighted {@link #weighted}
	 */
	public MultiVarKernelDensity(boolean weighted)
	{
		initialize(weighted);
	}
	
	/**
	 * @param weighted {@link #weighted}
	 */
	private void initialize(boolean weighted)
	{
		type				= MultiContProbDist.KERNEL;
		this.weighted		= weighted;
		samples				= new ArrayList<ContMultiSample>();
		statistics			= new ArrayList<ContStats>();
		scalarBandwidth		= Double.NaN;
		diagBandwidth		= null;
		active				= null;
		zeroVar				= null;
		linearlyDependent	= null;
		regressionCoeff		= null;
		invertible			= null;
		marginals			= null;
		minVariance			= DEF_MIN_VARIANCE;
	}
	
	// --------------------------------------------------------------------------------------------
	// Methods - Getters and setters
	// --------------------------------------------------------------------------------------------
	
	@Override
	public String getTypeString()
	{
		return ID;
	}

	@Override
	public int getDimensionality() 
	{
		return statistics.size();
	}

	/**
	 * @return {@link #weighted}
	 */
	public boolean isWeighted() 
	{
		return weighted;
	}

	/**
	 * Updates {@link #weighted} and computes {@link #statistics}
	 * @param weighted {@link #weighted}
	 * <p>Time complexity: <i>O(n*(d + k))</i>
	 * <br>Space complexity: <i>O(n*(d + k))</i>
	 * <br><i>n</i>: number of samples; <i>d</i>: number of dimensions; <i>k</i>: complexity
	 * of retrieving the weight and values from the {@link ContMultiSample} implementation
	 */
	public void setWeighted(boolean weighted) 
	{
		this.weighted = weighted;
		computeStatistics();
	}
	
	/**
	 * @return {@link #samples}
	 */
	public ArrayList<ContMultiSample> getSamples() 
	{
		return samples;
	}
	
	/**
	 * Returns a subset of the {@link samples} ordered by their weight. For unordered cases, or 
	 * when several samples have equal weights, the order is determined randomly between them.
	 * <p>Time complexity: <i>O(k)</i>
	 * <br>Space complexity: <i>O(1)</i>
	 * @param weightyFirst	True if the samples with the highest weight should be added first. 
	 * 						False if the ones with the lowest weight should be added first.
	 * @param weightPercent	The percentage of the total weight to be returned
	 * @param maximum		The maximum number of samples to be returned
	 * @return A list with the samples selected
	 */
	public ArrayList<ContMultiSample> getSamples(boolean weightyFirst, double weightPercent,
													int maximum)
	{
		ArrayList<ContMultiSample> result	= new ArrayList<>();
		int k								= samples.size();
		double weightSum					= 0.0;
		double totalWeight					= getWeightSum();
		if (weighted)
		{
			// Sort samples
			ArrayList<PointID> selector		= new ArrayList<>();
			for (int s = 0; s < samples.size(); s++)
			{
				ContMultiSample sample		= samples.get(s);
				selector.add(new PointID(s, sample.getWeight(), false));
			}
			Collections.shuffle(selector);
			Collections.sort(selector);
			
			// Select samples
			int s							= 0;
			while (result.size() < maximum && weightSum < weightPercent && s < k)
			{
				int index					= selector.get(s).getX();
				ContMultiSample sample		= samples.get(weightyFirst ? k - 1 - index : index);
				result.add(sample);
				s++;
				weightSum					+= sample.getWeight()/totalWeight;
			}
		}
		else
		{
			// Sort samples
			ArrayList<ContMultiSample> selector = new ArrayList<>();
			for (ContMultiSample sample : samples)
				selector.add(sample);
			Collections.shuffle(selector);
			
			// Select samples
			int s							= 0;
			while (result.size() < maximum && weightSum < weightPercent && s < k)
			{
				result.add(selector.get(s));
				s++;
				weightSum					= (double)s/k;
			}
		}
		return result;
	}
	
	/**
	 * Clears {@link #samples}. Also clears {@link #statistics}.
	 */
	public void clearSamples()
	{
		samples.clear();
		statistics.clear();
		scalarBandwidth		= Double.NaN;
		diagBandwidth		= null;
		active				= null;
		zeroVar				= null;
		linearlyDependent	= null;
		regressionCoeff		= null;
		invertible			= null;
		marginals			= null;
	}
	
	/**
	 * Adds a new sample to {@link #samples} and updates {@link #statistics}
	 * <p>Time complexity: <i>O(d + k)</i>
	 * <br>Space complexity: <i>O(d + k)</i>
	 * <br><i>d</i>: number of dimensions; <i>k</i>: complexity of retrieving the weight and values 
	 * from the {@link ContMultiSample} implementation
	 * @param sample The sample to add
	 */
	public void addSample(ContMultiSample sample)
	{
		for (int v = 0; v < sample.getValues().size(); v++)
			if (Double.isNaN(sample.getValues().get(v)))
				throw new IllegalArgumentException(
										String.format(ERR_VALUE_IS_NAN, v));
		samples.add(sample);
		updateStatistics(sample);
	}
	
	/**
	 * Sets {@link #samples} and computes {@link #statistics}
	 * <p>Time complexity: <i>O(n*(d + k))</i>
	 * <br>Space complexity: <i>O(n*(d + k))</i>
	 * <br><i>n</i>: number of samples; <i>d</i>: number of dimensions; <i>k</i>: complexity
	 * of retrieving the weight and values from the {@link ContMultiSample} implementation
	 * @param samples {@link #samples}
	 */
	public void setSamples(ArrayList<ContMultiSample> samples)
	{
		this.samples = samples;
		computeStatistics();
	}
	
	/**
	 * @return {@link #scalarBandwidth}
	 */
	public double getScalarBandwidth()
	{
		return scalarBandwidth;
	}

	/**
	 * @param bandwidth {@link #scalarBandwidth}
	 */
	public void setBandwidth(double bandwidth)
	{
		if (statistics.size() == 0)
			throw new RuntimeException(ERR_NO_SAMPLES);
		
		scalarBandwidth 	= bandwidth;
		diagBandwidth		= null;
		active				= new ArrayList<>();
		zeroVar				= new ArrayList<>();
		for (int d = 0 ; d < statistics.size(); d++)
			if (bandwidth > minVariance)
				active.add(d);
			else
				zeroVar.add(d);
		zeroVar				= null;
		linearlyDependent	= null;
		regressionCoeff		= null;
		invertible			= null;
		marginals			= null;
	}
	
	/**
	 * @return {@link #diagBandwidth}
	 */
	public double[] getDiagBandwidth()
	{
		return diagBandwidth;
	}

	/**
	 * @param bandwidths {@link #diagBandwidth}
	 */
	public void setBandwidth(double[] diagBandwidth)
	{
		if (statistics.size() == 0)
			throw new RuntimeException(ERR_NO_SAMPLES);
		if (diagBandwidth.length < statistics.size())
			throw new IllegalArgumentException(ERR_BW_VECTOR_SMALL + statistics.size());
			
		this.diagBandwidth	= diagBandwidth;
		active				= new ArrayList<>();
		zeroVar				= new ArrayList<>();
		for (int d = 0 ; d < statistics.size(); d++)
			if (diagBandwidth[d] >= minVariance)
				active.add(d);
			else
				zeroVar.add(d);
		
		scalarBandwidth		= Double.NaN;
		zeroVar				= null;
		linearlyDependent	= null;
		regressionCoeff		= null;
		invertible			= null;
		marginals			= null;
	}

	/**
	 * @param bandwidths {@link #diagBandwidth}
	 */
	public void setBandwidth(ArrayList<Double> bandwidths) 
	{
		setBandwidth(Utilities.toArray(bandwidths));
	}
	
	/**
	 * @return A covariance matrix that represents the extent and orientation of the kernels. If 
	 * only the main diagonal is non-zero, with a single global value, the kernels are of 
	 * <i>S</i>-class. If only the diagonal is non-zero, but the elements are different, the 
	 * kernels are <i>D</i>-class; if the matrix contains non-zero elements outside of the main
	 * diagonal, the kernels are <i>F</i>-class.
	 * <p>Time complexity: <i>O(k^2)</i>
	 * <br>Space complexity: <i>O(k^2)</i>
	 */
	public double[][] getBandwidth()
	{
		try
		{
			int k				= statistics.size();
			double [][] bw		= new double[k][k];
			if (!Double.isNaN(scalarBandwidth))
				for (int d = 0; d < k; d++)
					bw[d][d]	= scalarBandwidth;
			else if (diagBandwidth != null)
				for (int d = 0; d < k; d++)
					bw[d][d]	= diagBandwidth[d];
			else if (active != null)
			{
				for (int a = 0; a < active.size(); a++)
				{
					int d		= active.get(a);
					for (int a2 = 0; a2 < active.size(); a2++)
					{
						int d2	= active.get(a2);
						bw[d][d2]	= invertible.getC()[a][a2];
					}
					
					// TODO Include coefficients for linearly dependent variables
				}
			}
			else
				throw new RuntimeException(ERR_BW_NOT_DEFINED);
			return bw;
		} catch (Exception e)
		{
			RuntimeException e2	= new RuntimeException(ERR_MATRIX_TOO_BIG + e.getMessage());
			e2.initCause(e);
			throw e2;
		}
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
	
	// --------------------------------------------------------------------------------------------
	// Methods - Computations
	// --------------------------------------------------------------------------------------------
	
	/**
	 * Clears {@link #statistics} and re-fills it with the data in {@link #samples}. Call this 
	 * method when the actual samples or their weights have been modified externally.
	 * <p>Time complexity: <i>O(n*(d + k))</i>
	 * <br>Space complexity: <i>O(n*(d + k))</i>
	 * <br><i>n</i>: number of samples; <i>d</i>: number of dimensions; <i>k</i>: complexity
	 * of retrieving the weight and values from the {@link ContMultiSample} implementation
	 */
	public void computeStatistics()
	{
		statistics.clear();
		for (ContMultiSample sample : samples)
			updateStatistics(sample);
	}

	/**
	 * Updates {@link #statistics} with a new sample
	 * <p>Time complexity: <i>O(d + k)</i>
	 * <br>Space complexity: <i>O(d + k)</i>
	 * <br><i>d</i>: number of dimensions; <i>k</i>: complexity of retrieving the weight and values 
	 * from the {@link ContMultiSample} implementation
	 * @param sample The sample to add
	 */
	private void updateStatistics(ContMultiSample sample) 
	{
		ArrayList<Double> values	= sample.getValues();
		int statsSize 				= statistics.size();
		int sampleSize				= values.size();
		if (statsSize < sampleSize)
			for (int size = statsSize + 1; size <= sampleSize; size++)
				statistics.add(new ContStats(weighted));
		double weight	= sample.getWeight();
		for (int i = 0; i < sampleSize; i++)
		{
			double value	= values.get(i);
			if (!Double.isNaN(value))
				statistics.get(i).addValue(value, weight);
		}
		
		scalarBandwidth				= Double.NaN;
		diagBandwidth				= null;
		active						= null;
		zeroVar						= null;
		linearlyDependent			= null;
		regressionCoeff				= null;
		invertible					= null;
		marginals					= null;
	}
	
	/**
	 * @return The sum of the weights of all the samples
	 */
	public double getWeightSum()
	{
		return statistics.get(0).getWeightSum();
	}
	
	/**
	 * @return A list with the marginal mean values of the distribution for every dimension
	 */
	public ArrayList<Double> getMeanAL()
	{
		ArrayList<Double> means	= new ArrayList<Double>(statistics.size());
		for (ContStats dimension : statistics)
			means.add(dimension.getMean());
		return means;
	}
	
	@Override
	public double[] getMean() 
	{
		ArrayList<Double> meanAL	= getMeanAL();
		double[] mean				= new double[meanAL.size()];
		for (int m = 0; m < meanAL.size(); m++)
			mean[m]					= meanAL.get(m);
		return mean;
	}
	
	@Override
	public double[] getVariance()
	{
		int dimensions	= getDimensionality();
		double[] var	= new double[dimensions];
		if (!Double.isNaN(scalarBandwidth))
		{
			for (int d = 0; d < dimensions; d++)
				var[d]		= scalarBandwidth + statistics.get(d).getVar();
			return var;
		}
		if (diagBandwidth != null)
		{
			for (int d = 0; d < dimensions; d++)
				var[d]		= diagBandwidth[d] + statistics.get(d).getVar();
			return var;
		}
		if (active != null)
		{
			for (Integer d: zeroVar)
				var[d]		= 0.0;
			for (Integer d: active)
				var[d]		= invertible.getC()[d][d] + statistics.get(d).getVar();
			for (Integer d: linearlyDependent)
			{
				double sum	= 0.0;
				for (PointID term: regressionCoeff.get(d))
				{
					if (term.x == -2)
						sum	+= term.y;
					else if (term.x >= 0)
					{
						double vard2 = statistics.get(term.x).getVar();
						sum	+= term.y*term.y*vard2*vard2;
					}
				}
				var[d]		= sum;
			}
			return var;
		}
		else
			throw new RuntimeException(ERR_BW_NOT_DEFINED);
	}
	
	@Override
	public double[][] getCovariance() 
	{
		// TODO This method might be wrong: it needs to implement the law of total covariance right 
		
		double[][] covariance			= getBandwidth();
		int k							= statistics.size();
		double[] variance				= getVariance();
		for		(int d1 = 0;	d1 < k; d1++)
			for	(int d2 = d1;	d2 < k; d2++)
				if (d1 == d2)
					covariance[d1][d2]	= variance[d1];
				else
				{
					double cov			= computeCovariance(d1, d2, 0.0);
					covariance[d1][d2]	+= cov;
					covariance[d2][d1]	+= cov;
				}
		return covariance;
	}
	
	/**
	 * Estimates the bandwidth of each dimension (D-class kernels) using Silverman's rule for 
	 * univariate distributions. The coefficient for the bandwidth (<i>h</i>) matrix 
	 * is computed as: <i>h = [(4*sig^5)/(3*n)]^(2/5)</i>, where <i>sig</i> is the standard
	 * deviation of the samples for that dimension, and <i>n</i> is the number of samples.
	 * <p>Time complexity: <i>O(number of dimensions)</i>
	 * <br>Space complexity: <i>O(number of dimensions)</i>
	 */
	public void silvermanUnivarDiagBW()
	{
		diagBandwidth			= new double[statistics.size()];
		int sampleCount			= samples.size();
		int dimensions			= statistics.size();
		double temp1			= 1.059223841*Math.pow(sampleCount, -1.0/5);
		active					= new ArrayList<>();
		zeroVar					= new ArrayList<>();
		for (int i = 0; i < dimensions; i++)
		{
			ContStats series	= statistics.get(i);
			double stDev		= series.getStDev();
			double temp2		= stDev*temp1;
			diagBandwidth[i] 	= temp2*temp2;
			if (diagBandwidth[i] >= minVariance)
				active.add(i);
			else
				zeroVar.add(i);
		}
		linearlyDependent		= null;
		regressionCoeff			= null;
		invertible				= null;
		marginals				= null;
	}
	
	/**
	 * Estimates the bandwidth of each dimension (D-class kernels) using the optimal criteria for 
	 * an assumed underlying Gaussian distributions as described by Scott's rule.
	 * <p>Time complexity: <i>O(number of dimensions)</i>
	 * <br>Space complexity: <i>O(number of dimensions)</i>
	 */
	public void scottDiagBW()
	{
		computeGaussianDiagBW(false);
	}

	/**
	 * Estimates the bandwidth of each dimension (D-class kernels) using the optimal criteria for 
	 * an assumed underlying Gaussian distributions as described in the reference below. The 
	 * criteria is termed "normal distribution approximation" or "Silverman's rule of thumb". 
	 * <br><br>
	 * Silverman, B.W., 1998, <i>Density Estimation for Statistics and Data Analysis</i>, London: 
	 * Chapman & Hall/CRC, p.48, ISBN 0-412-24620-1.
	 * <p>Time complexity: <i>O(number of dimensions)</i>
	 * <br>Space complexity: <i>O(number of dimensions)</i>
	 */
	public void silvermanDiagBW()
	{
		computeGaussianDiagBW(true);
	}

	/**
	 * Estimates the bandwidth of each dimension using either Scott's rule
	 * ({@link #scottDiagBW()}) or Silverman's rule ({@link #silvermanDiagBW()})
	 * <p>Time complexity: <i>O(number of dimensions)</i>
	 * <br>Space complexity: <i>O(number of dimensions)</i>
	 * @param silvermansRule True if Silverman's rule should be used; false if Scott's rule should
	 * be used
	 */
	public void computeGaussianDiagBW(boolean silvermansRule)
	{
		double scaling			= silvermansRule ? getSilvermansScaling() : getScottsScaling();
		computeGaussianDiagBW(scaling);
	}
	
	/**
	 * Estimates the bandwidth of each dimension using either Scott's rule
	 * ({@link #scottDiagBW()}) or Silverman's rule ({@link #silvermanDiagBW()})
	 * <p>Time complexity: <i>O(number of dimensions)</i>
	 * <br>Space complexity: <i>O(number of dimensions)</i>
	 * @param scaling	The scalar factor to multiply the covariance of each dimensions to obtain
	 * 					the	bandwidth vector
	 */
	public void computeGaussianDiagBW(double scaling)
	{
		diagBandwidth			= new double[statistics.size()];
		int dimensions			= statistics.size();
		active					= new ArrayList<>();
		zeroVar					= new ArrayList<>();
		for (int i = 0; i < dimensions; i++)
		{
			ContStats series	= statistics.get(i);
			double var			= series.getVar();
			diagBandwidth[i]	= var*scaling;
			if (var >= minVariance)
				active.add(i);
			else
				zeroVar.add(i);
		}
		linearlyDependent		= null;
		regressionCoeff			= null;
		invertible				= null;
		marginals				= null;
	}
	
	/**
	 * @param bandwidth A covariance matrix that represents the extent and orientation of the 
	 * kernels. If only the main diagonal is non-zero, with a single global value, the kernels are 
	 * of <i>S</i>-class. If only the diagonal is non-zero, but the elements are different, the 
	 * kernels are <i>D</i>-class; if the matrix contains non-zero elements outside of the main
	 * diagonal, the kernels are <i>F</i>-class. The matrix must be symmetric and positive 
	 * definite. For improved efficiency, {@link #scalarBandwidth} or {@link #diagonalBandwidth}
	 * should be used to define the bandwidth instead if <i>F</i>-class kernels are not to be used.
	 */
	public void setBandwidth(double[][] bandwidth)
	{
		// Initial verifications
		RealMatrix bandwidthMat		= new Array2DRowRealMatrix(bandwidth);
		int rows					= bandwidthMat.getRowDimension();
		int cols					= bandwidthMat.getColumnDimension();
		int k						= statistics.size();
		if (rows < k || cols < k)
		{
			String error	= ERR_BW_MATRIX_SMALL + k + " x " + k;
			throw new IllegalArgumentException(error);
		}
		if (!MatrixUtils.isSymmetric(bandwidthMat, 1E-3))
			throw new IllegalArgumentException(ERR_NOT_SYMMETRIC);
		
		// Prepare data structures
		scalarBandwidth				= Double.NaN;
		diagBandwidth				= null;
		marginals					= null;
		active						= new ArrayList<>(k);
		zeroVar						= new ArrayList<>(k);
		linearlyDependent			= new ArrayList<>(k);
		regressionCoeff				= new Hashtable<>(k);
		invertible					= new MultiVarNormal(new double[0], new double[0]);
		
		// Determine zero-variance dimensions
		ArrayList<Integer> candidates = new ArrayList<>();
		for (int d = 0; d < k; d++)
			if (statistics.get(d).getVar() >= minVariance)
				candidates.add(d);
			else
				zeroVar.add(d);
		
		if (candidates.size() == 0)
		{
			scalarBandwidth			= 0.0;
			linearlyDependent		= null;
			regressionCoeff			= null;
			invertible				= null;
			return;
		}
		
		// Add first element
		int d						= candidates.get(0);
		double[][] m				= {{bandwidth[d][d]}};
		RealMatrix mat1				= new Array2DRowRealMatrix(m);
		active.add(d);
		
		regressionCoeff				= new Hashtable<>(k);
		for (int i = 1; i < candidates.size(); i++)
		{
			d						= candidates.get(i);
			int size				= active.size();
			double[] v				= null;
			
			// Prepare extended matrix
			try
			{
				RealMatrix mat2		= new Array2DRowRealMatrix(new double[size + 1][size + 1]);
				mat2.setSubMatrix(mat1.getData(), 0, 0);
				v					= new double[size];
				for (int c = 0; c < size; c++)
				{
					int d2			= active.get(c);
					double cov		= bandwidth[d][d2];
					v[c]			= cov;
					mat2.setEntry(size, c, cov);
					mat2.setEntry(c, size, cov);
				}
				mat2.setEntry(size, size, bandwidth[d][d]);
				
				try
				{
					// Compute the determinant of the extended matrix
					double[][] low	= MatUtil.cholesky(mat2.getData());
					double detL		= MatUtil.getDeterminantTriangular(low);
					double determinant = detL*detL;
					
					// Determine if the extended matrix is invertible
					if (!Double.isNaN(determinant) && determinant >= INVERTIBLE_THRESHOLD)
					{
						active.add(d);
						mat1			= mat2;
						continue;
					}
				} catch (Exception e) {}
			}
			catch (Exception e)	{}
			
			// Compute new covariance values if missing
			if (v == null)
			{
				v					= new double[size];
				for (int c = 0; c < size; c++)
					v[c]			= bandwidth[d][active.get(c)];
			}
			
			// Mark as dependent and compute regression coefficients
			linearlyDependent.add(d);
			double[] coeffs			= MatUtil.luEvaluate(mat1.getData(), v);
			RealVector coeffsV		= new ArrayRealVector(coeffs);
			
			// Compute constant vector
			ContStats values		= statistics.get(d);
			double[] means			= new double[size];
			for (int c = 0; c < size; c++)
				means[c]			= statistics.get(active.get(c)).getMean();
			double combination		= coeffsV.dotProduct(new ArrayRealVector(means));
			double constant			= values.getMean() - combination;
			
			// Save coefficients
			ArrayList<PointID> reg	= new ArrayList<>(2 + size);
			reg.add(new PointID(-1, constant));
			for (int c = 0; c < size; c++)
				if (coeffs[c] != 0.0)
					reg.add(new PointID(active.get(c), coeffs[c]));
			regressionCoeff.put(d, reg);
		}
		
		// Create kernel with invertible covariance matrix
		if (active.size() > 0 && active.size() < statistics.size() && bandwidthMat != null)
		{
			double[] zeroVec		= new double[active.size()];
			invertible				= new MultiVarNormal(zeroVec, mat1.getData());
			invertible.markCovarianceAsInvertible();
		}
		else
			invertible				= null;
		
		// Log results
		if (k > 10)
			System.out.println("Kernel density distribution: " + zeroVar.size() + " constants; " 
								+ invertible.getDimensionality() + " covariates; " 
								+ linearlyDependent.size() + " linearly-dependent");
	}
	
	/**
	 * Computes {@link #invertible}. Variables with zero variance ({@link #zeroVar}) or that are 
	 * linearly dependent on others ({@link #linearlyDependent} and {@link #regressionCoeff}) are 
	 * identified and isolated. The bandwidth matrix is computed as the sample covariance matrix 
	 * times a scaling factor. The scaling factor uses either Scott's rule ({@link #scottDiagBW()})
	 * or Silverman's rule ({@link #silvermanDiagBW()}).
	 * @param silvermansRule	True if Silverman's rule should be used; false if Scott's rule 
	 * 							should be used
	 */
	public void computeGaussianBW(boolean silvermansRule)
	{
		computeGaussianBW(silvermansRule, Integer.MAX_VALUE, 0.0);
	}
	
	/**
	 * Computes {@link #invertible}. Variables with zero variance ({@link #zeroVar}) or that are 
	 * linearly dependent on others ({@link #linearlyDependent} and {@link #regressionCoeff}) are 
	 * identified and isolated. The bandwidth matrix is computed as the sample covariance matrix 
	 * times a scaling factor.
	 * @param scaling	The scalar factor to multiply the sample covariance matrix to obtain the
	 * 					bandwidth matrix
	 */
	public void computeGaussianBW(double scaling)
	{
		computeGaussianBW(scaling, Integer.MAX_VALUE, 0.0);
	}
	
	/**
	 * Computes {@link #invertible}. Variables with zero variance ({@link #zeroVar}) or that are 
	 * linearly dependent on others ({@link #linearlyDependent} and {@link #regressionCoeff}) are 
	 * identified and isolated. The bandwidth matrix is computed as the sample covariance matrix 
	 * times a scaling factor. The scaling factor uses either Scott's rule ({@link #scottDiagBW()})
	 * or Silverman's rule ({@link #silvermanDiagBW()}). The size of the matrix can also be 
	 * arbitrarily limited, and made sparse by ignoring weak dependencies.
	 * @param silvermansRule	True if Silverman's rule should be used; false if Scott's rule 
	 * 							should be used
	 * @param dimLimit			The maximum size of the covariance matrix
	 * @param corrThreshold		If the absolute value of any computed correlation is smaller than
	 * 							this threshold, it is assigned a value of zero. That is, use values
	 * 							larger than zero to impose sparsity on the covariance matrix.
	 */
	public void computeGaussianBW(boolean silvermansRule, int dimLimit, double corrThreshold)
	{
		double scaling = silvermansRule	? getSilvermansScaling() : getScottsScaling();
		computeGaussianBW(scaling, dimLimit, corrThreshold);
	}

	/**
	 * Computes {@link #invertible}. Variables with zero variance ({@link #zeroVar}) or that are 
	 * linearly dependent on others ({@link #linearlyDependent} and {@link #regressionCoeff}) are 
	 * identified and isolated. The bandwidth matrix is computed as the sample covariance matrix 
	 * times a scaling factor. The size of the matrix can also be arbitrarily limited, and made
	 * sparse by ignoring weak dependencies.
	 * @param scaling		The scalar factor to multiply the sample covariance matrix to obtain
	 * 						the bandwidth matrix 
	 * @param dimLimit		The maximum size of the covariance matrix
	 * @param corrThreshold	If the absolute value of any computed correlation is smaller than this
	 * 						threshold, it is assigned a value of zero. That is, use values larger
	 * 						than zero to impose sparsity on the covariance matrix.
	 */
	public void computeGaussianBW(double scaling, int dimLimit, double corrThreshold)
	{
		int dimensions					= statistics.size();
		if (dimensions == 0)
			throw new RuntimeException(ERR_NO_SAMPLES);
		
		// Initialize structures
		active							= new ArrayList<>(dimensions);
		zeroVar							= new ArrayList<>(dimensions);
		linearlyDependent				= new ArrayList<>(dimensions);
		invertible						= null;
		int semiDependents				= 0;
		
		// Determine zero-variance dimensions
		ArrayList<Integer> candidates	= new ArrayList<>();
		for (int d = 0; d < dimensions; d++)
			if (statistics.get(d).getVar() >= minVariance)
				candidates.add(d);
			else
				zeroVar.add(d);
		
		if (candidates.size() == 0)
		{
			scalarBandwidth			= 0.0;
			linearlyDependent		= null;
			regressionCoeff			= null;
			invertible				= null;
			return;
		}
		
		// Prepare arrays and randomize candidate dimensions
		ArrayList<Integer> act			= new ArrayList<>(candidates.size()	);
		ArrayList<Integer> dep			= new ArrayList<>(dimensions		);
		ArrayList<Integer> order		= new ArrayList<>(candidates		);
		Collections.shuffle(order);
		
		// Add first element
		int first						= order.get(0);
		double[][] m					= {{scaling*statistics.get(first).getVar()}};
		RealMatrix mat1					= new Array2DRowRealMatrix(m);
		act.add(first);
		
		regressionCoeff					= new Hashtable<>(dimensions);
		for (int i = 1; i < order.size(); i++)
		{
			int d						= order.get(i);
			int size					= act.size();
			double[] covs				= null;
			if (size < dimLimit)
			{
				// Prepare extended matrix
				try
				{
					RealMatrix mat2		= new Array2DRowRealMatrix(new double[size + 1][size + 1]);
					covs				= new double[size];
					
					// Find insertion point
					int index			= 0;
					for (index = 0; index <= act.size(); index++)
						if (index == size)
							break;
						else if (d < act.get(index))
							break;
										
					// Set new values
					mat2.setEntry(index, index, scaling*statistics.get(d).getVar());
					for (int c = 0; c < size; c++)
					{
						int d2			= act.get(c);
						double cov		= scaling*computeCovariance(d, d2, corrThreshold);
						covs[c]			= cov;
						int index2		= c < index ? c : c + 1; 
						mat2.setEntry(index, index2, cov);
						mat2.setEntry(index2, index, cov);
					}
					
					// Set values from original matrix
					if (index == 0)
						mat2.setSubMatrix(mat1.getData(), 1, 1);
					else if (index == size)
						mat2.setSubMatrix(mat1.getData(), 0, 0);
					else
					{
						RealMatrix sub	= mat1.getSubMatrix(0, index - 1, 0, index - 1);
						mat2.setSubMatrix(sub.getData(), 0, 0);
						sub				= mat1.getSubMatrix(index, size - 1, index, size - 1);
						mat2.setSubMatrix(sub.getData(), index + 1, index + 1);
						sub				= mat1.getSubMatrix(0, index - 1, index, size - 1);
						mat2.setSubMatrix(sub.getData(), 0, index + 1);
						sub				= mat1.getSubMatrix(index, size - 1, 0, index - 1);
						mat2.setSubMatrix(sub.getData(), index + 1, 0);
					}
					
					// Compute the determinant of the extended matrix
					try
					{
						double[][] low	= MatUtil.cholesky(mat2.getData());
						double detL		= MatUtil.getDeterminantTriangular(low);
						double det		= detL*detL;
						
						// Determine if the extended matrix is invertible
						if (!Double.isNaN(det) && det >= INVERTIBLE_THRESHOLD)
						{
							act.add(d);
							Collections.sort(act);
							mat1			= mat2;
							continue;
						}
					} catch (Exception e) {}
				}
				catch (Exception e)	{}
			}
			
			// Compute new covariance values if missing
			if (covs == null)
			{
				covs					= new double[size];
				for (int c = 0; c < size; c++)
					covs[c]				= computeCovariance(d, act.get(c), corrThreshold);
			}
			
			// Mark as dependent and compute regression coefficients
			dep.add(d);
			double[][] lower			= MatUtil.cholesky(mat1.getData());
			double[] coeffs				= MatUtil.luEvaluate(lower, covs);
			RealVector coeffsV			= new ArrayRealVector(coeffs);
			
			// Compute constant vector
			ContStats values			= statistics.get(d);
			double[] means				= new double[size];
			for (int c = 0; c < size; c++)
				means[c]				= statistics.get(act.get(c)).getMean();
			double combination			= coeffsV.dotProduct(new ArrayRealVector(means));
			double constant				= values.getMean() - combination;
			
			// Compute independent variance
			double independentVar		= 0.0;
			if (mat1.getRowDimension() >= dimLimit)
			{
				double totalVar			= values.getVar();
				double explainedVar		= 0.0;
				for 	(int c1 = 0; c1 < size; c1++)
				{
					double coeff1		= coeffs[c1];
					explainedVar		+= coeff1*coeff1*mat1.getData()[c1][c1];
					for	(int c2 = c1 + 1; c2 < size; c2++)
					{
						double coeff2	= coeffs[c2];
						explainedVar	+= 2*coeff1*coeff2*mat1.getData()[c1][c2];
					}
				}
				independentVar			= Math.max(0.0, totalVar - explainedVar);
			}
			
			// Save coefficients
			ArrayList<PointID> reg		= new ArrayList<>(2 + size);
			reg.add(new PointID(-1, constant));
			if (independentVar > 0.0)
			{
				reg.add(new PointID(-2, independentVar));
				semiDependents++;
			}
			for (int c = 0; c < size; c++)
				if (coeffs[c] != 0.0)
					reg.add(new PointID(act.get(c), coeffs[c]));
			regressionCoeff.put(d, reg);
		}
		
		int size						= act.size();
		if (size > 0)
		{
			// Sort dependent variables and create kernel
			active						= new ArrayList<>(act);
			linearlyDependent			= new ArrayList<>(dep);
			Collections.sort(linearlyDependent);
			double[] zeroVec			= new double[size];
			invertible					= new MultiVarNormal(zeroVec, mat1.getData());
			invertible.markCovarianceAsInvertible();
		}
		
		// Log results
		if (dimensions > 10)
		{
			String line					= "Kernel density distribution: " + zeroVar.size() 
												+ " constants; " + invertible.getDimensionality() 
												+ " covariates; " + linearlyDependent.size();
			line						+= semiDependents > 0 ? (" (" + semiDependents 
								+ ") linearly-dependent (partially)") : (" linearly-dependent")
								+ "; scaling = " + scaling;
			System.out.println(line);
		}
	}
	
	/**
	 * @param dim1			The index of the first dimension
	 * @param dim2			The index of the second dimension
	 * @param corrThreshold	If the absolute value of any computed correlation is smaller than
	 * 						this threshold, it is assigned a value of zero. That is, use values
	 * 						larger than zero to impose independence between variables.
	 * @return		The covariance of the two dimensions specified
	 */
	private double computeCovariance(int dim1, int dim2, double corrThreshold)
	{
		ContPairedSeries series	= new ContPairedSeries();
		for (ContMultiSample sample : samples)
			series.addPair(sample.getValues().get(dim1), sample.getValues().get(dim2));
		double absCorr			= Math.abs(series.getCorrel());
		return absCorr < corrThreshold ? 0.0 : series.getCov();
	}
	
	/**
	 * @return The scaling factor for the bandwidth matrix according to Silverman's rule for 
	 * assumed underlying Gaussian distributions as described in the reference below. The criteria 
	 * is termed "normal distribution approximation" or "Silverman's rule of thumb".
	 * <br><br>
	 * Silverman, B.W., 1998, <i>Density Estimation for Statistics and Data Analysis</i>, London: 
	 * Chapman & Hall/CRC, p.48, ISBN 0-412-24620-1.
	 * <p>Time complexity: <i>O(1)</i>
	 * <br>Space complexity: <i>O(1)</i>
	 */
	public double getSilvermansScaling()
	{
		int samples		= this.samples.size();
		int dimensions	= statistics.size();
		return getSilvermansScaling(samples, dimensions);
	}
	
	/**
	 * Returns the scaling factor for the bandwidth matrix according to Silverman's rule for 
	 * assumed underlying Gaussian distributions as described in the reference below. The criteria
	 * is termed "normal distribution approximation" or "Silverman's rule of thumb". <br><br>
	 * Silverman, B.W., 1998, <i>Density Estimation for Statistics and Data Analysis</i>, London: 
	 * Chapman & Hall/CRC, p.48, ISBN 0-412-24620-1.
	 * <p>Time complexity: <i>O(1)</i>
	 * <br>Space complexity: <i>O(1)</i>
	 * @param samples		The number of samples
	 * @param dimensions	The number of dimensions
	 * @return The scaling factor for the bandwidth matrix according to Silverman's rule
	 */
	public static double getSilvermansScaling(int samples, int dimensions)
	{
		double temp1	= Math.pow(samples, -1.0/(dimensions + 4));
		double temp2	= Math.pow(4.0/(dimensions + 2), 1.0/(dimensions + 4));
		return temp1*temp1*temp2*temp2;
	}
	
	/**
	 * @return The scaling factor for the bandwidth matrix according to Scott's rule for assumed 
	 * underlying Gaussian distributions
	 * <p>Time complexity: <i>O(1)</i>
	 * <br>Space complexity: <i>O(1)</i>
	 */
	public double getScottsScaling()
	{
		int samples		= this.samples.size();
		int dimensions	= statistics.size();
		return getScottsScaling(samples, dimensions);
	}
	
	/**
	 * Returns the scaling factor for the bandwidth matrix according to Scott's rule for assumed 
	 * underlying Gaussian distributions
	 * <p>Time complexity: <i>O(1)</i>
	 * <br>Space complexity: <i>O(1)</i>
	 * @param samples		The number of samples
	 * @param dimensions	The number of dimensions
	 * @return The scaling factor for the bandwidth matrix according to Scott's rule
	 */
	public static double getScottsScaling(int samples, int dimensions)
	{
		double temp1	= Math.pow(samples, -1.0/(dimensions + 4));
		return temp1*temp1;
	}

	@Override
	public double getpdf(double[] x)
	{
		int k								= statistics.size();
		if (k == 0) 				
			throw new RuntimeException(ERR_NO_SAMPLES);
		if (x.length < k)
			throw new IllegalArgumentException(String.format(ERR_PARTICLE_SIZE, x.length, k));
		
		// S-class kernel case
		if (!Double.isNaN(scalarBandwidth))
			return computepdfSKernels(x);
		
		double pdf							= 1.0;
		if (active.size() < k)
		{
			// Verify zero variance dimensions
			for (Integer dim : zeroVar)
				if (Math.abs(x[dim] - statistics.get(dim).getMean()) > Math.sqrt(minVariance))
					return 0.0;
				
			// Verify linearly-dependent dimensions
			for (int d = 0; d < linearlyDependent.size(); d++)
			{
				Integer dim				= linearlyDependent.get(d);
				ArrayList<PointID> reg	= regressionCoeff.get(dim);
				double target			= 0.0;
				double residVar			= minVariance;
				for (int t = 0; t < reg.size(); t++)
				{
					PointID point		= reg.get(t);
					int index			= point.x;
					double coeff		= point.y;
					if (index == -2)
						residVar		= coeff;
					else if (index == -1)
						target			+= coeff;
					else
						target			+= x[index]*coeff;
				}
				pdf						*= Normal.computepdf(target, Math.sqrt(residVar), x[dim]);
				if (pdf == 0.0)
					return 0.0;
			}
			
			// Prepare truncated particle
			double[] xTrunc				= new double[active.size()];
			for (int d = 0; d < active.size(); d++)
				xTrunc[d]				= x[active.get(d)];
			x							= xTrunc;
		}
		
		// Compute density for D-class or F-class kernels
		if (diagBandwidth != null)
			return pdf*computepdfDKernels(x);
		else
			return pdf*computepdfFKernels(x);
	}

	@Override
	public double getpdf(ArrayList<Double> x)
	{
		return getpdf(Utilities.toArray(x));
	}
	
	@Override
	public double getLogpdf(double[] x)
	{
		return Math.log(getpdf(x));
	}

	@Override
	public double getLogpdf(ArrayList<Double> x)
	{
		return Math.log(getpdf(x));
	}
	
	/**
	 * Computes the value of the probability density function at the provided point for 
	 * <i>S</i>-class kernels. The density is the weighted average of the kernel density around 
	 * each of the samples.
	 * @param x	The point at which to evaluate the probability density function
	 * @return	The value of the probability density function at the provided point
	 */
	private double computepdfSKernels(double[] x)
	{
		int k						= statistics.size();
		double[] partArr			= new double[k];
		double[] variance			= new double[k];
		for (int i = 0; i < k; i++)
		{
			partArr[i]				= x[i];
			variance[i]				= scalarBandwidth;
		}
		double sum					= 0;
		double weightSum			= 0;
		MultiVarNormal kernel		= new MultiVarNormal(new double[k], variance);
		kernel.markCovarianceAsInvertible();
		for (ContMultiSample sample : samples)
		{
			double[] mean			= new double[k];
			for (int i = 0; i < k; i++)
				mean[i]				= sample.getValues().get(i);
			kernel.setMean(mean);
			double pdfSample		= kernel.getpdf(partArr);
			double weight			= weighted ? sample.getWeight() : 1.0;
			sum						+= weight*pdfSample;
			weightSum				+= weight;
		}
		return sum/weightSum;
	}

	/**
	 * Computes the value of the probability density function at the provided point for 
	 * <i>D</i>-class kernels. The density is the weighted average of the kernel density around 
	 * each of the samples.
	 * @param x	The point at which to evaluate the probability density function
	 * @return	The value of the probability density function at the provided point
	 */
	private double computepdfDKernels(double[] x)
	{
		int k						= active.size();
		double sum					= 0;
		double weightSum			= 0;
		MultiVarNormal kernel		= k < statistics.size() ? invertible
												: new MultiVarNormal(new double[k], diagBandwidth);
		for (ContMultiSample sample : samples)
		{
			double[] mean			= new double[k];
			for (int i = 0; i < k; i++)
				mean[i]				= sample.getValues().get(active.get(i));
			kernel.setMean(mean);
			double pdfSample		= kernel.getpdf(x);
			double weight			= weighted ? sample.getWeight() : 1.0;
			sum						+= weight*pdfSample;
			weightSum				+= weight;
		}
		return sum/weightSum;
	}

	/**
	 * Computes the value of the probability density function at the provided point for 
	 * <i>F</i>-class kernels. The density is the weighted average of the kernel density around 
	 * each of the samples. Equations found at:
	 * http://stats.stackexchange.com/questions/147210/efficient-fast-mahalanobis-distance-computation
	 * @param x	The point at which to evaluate the probability density function
	 * @return	The value of the probability density function at the provided point
	 */
	private double computepdfFKernels(double[] x)
	{
		int k						= active.size();
		double[][] matrix			= invertible.getC();
		double[][] lower			= null;
		double detB					= Double.NaN;
		lower						= MatUtil.cholesky(matrix);
		double detL					= MatUtil.getDeterminantTriangular(lower);
		detB						= detL*detL;
		double sum					= 0;
		double weightSum			= 0;
		for (ContMultiSample sample : samples)
		{
			double[] dev			= new double[k];
			for (int i = 0; i < k ; i++)
				dev[i]				= x[i] - sample.getValues().get(active.get(i));
			double[] y				= MatUtil.solveLower(lower, dev);
			double exp				= 0.0;
			for (int i = 0; i < y.length; i++)
				exp					+= y[i]*y[i];
			exp						/= -2.0;
			double weight			= sample.getWeight();
			double contrib			= Math.exp(exp);
			sum						+= weighted ? weight*contrib : contrib;
			weightSum				+= weighted ? weight : 1.0;
		}
		double normalizer			= weightSum*Math.sqrt(Math.pow(2*Math.PI, k)*detB);
		return Double.isNaN(normalizer) || Double.isInfinite(sum)	? Double.POSITIVE_INFINITY
																	: sum/normalizer;
	}
	
	@Override
	public double getMeanIndeppdf(double[] x)
	{
		int k								= statistics.size();
		if (k == 0) 						throw new RuntimeException(ERR_NO_SAMPLES);
		if (x.length == 0)					return Double.NaN;
		
		// Obtain marginal distributions
		if (marginals == null)
		{
			marginals						= new ArrayList<>();
			synchronized (marginals)
			{
				marginals 					= marginalizeMultiple(active);
			}
		}
		
		synchronized (marginals)
		{
			if (marginals.size() == 0)
				return Double.NaN;
			
			// Compute mean standard marginal likelihood
			ContStats series				= new ContStats(false);
			for (int i = 0; i < active.size(); i++)
			{
				int dimension				= active.get(i);
				KernelDensity marginal		= marginals.get(i);
				double value				= x[dimension];
				if (!Double.isNaN(value) && marginal != null)
				{
					double margLikelihood	= marginal.getpdf(value);
					double standardized		= margLikelihood*marginal.getStDev();
					series.addValue(standardized);
				}
			}
			return series.getMean();
		}
	}
	
	/**
	 * Computes the average independent standardized probability density function (pdf, or 
	 * likelihood) value at the provided point. That is, first marginalizes the distribution for 
	 * each dimension, then computes the standardized pdf value of each marginalized distribution, 
	 * and then computes the average of the individual independent values. The standardized pdf is 
	 * the usual pdf multiplied by the distribution's standard deviation. This is analogous to 
	 * first standardizing the samples and then computing the pdf, and is done so that the pdf
	 * values of the marginal distributions are comparable between themselves independently of 
	 * their spread.
	 * @param x	The point at which to evaluate the average independent standardized 
	 * 			probability density function
	 * @return	The average independent standardized probability density function value. 
	 * 			{@link java.lang.Double#NaN} if all dimensions have zero variance.
	 */
	public double getMeanIndpdf(ArrayList<Double> x)
	{
		return getMeanIndeppdf(Utilities.toArray(x));
	}
	
	/**
	 * Computes the average independent standardized probability density function (pdf, or 
	 * likelihood) value at the provided point. That is, first marginalizes the distribution for 
	 * each dimension, then computes the standardized pdf value of each marginalized distribution, 
	 * and then computes the average of the individual independent values. The standardized pdf is 
	 * the usual pdf multiplied by the distribution's standard deviation. This is analogous to 
	 * first standardizing the samples and then computing the pdf, and is done so that the pdf
	 * values of the marginal distributions are comparable between themselves independently of 
	 * their spread.
	 * @param particle The point at which to evaluate the average independent standardized 
	 * probability density function
	 * @return The average independent standardized probability density function value
	 */
	public ArrayList<Double> getMeanIndpdfMultiple(ArrayList<ArrayList<Double>> points)
	{
		int k						= statistics.size();
		if (k == 0) 				throw new RuntimeException(ERR_NO_SAMPLES);
		ArrayList<Double> result	= new ArrayList<>();
		for (ArrayList<Double> x : points)
			result.add(x.size() < k ? Double.NaN : getMeanIndpdf(x));
		return result;
	}
	
	@Override
	public double getMeanMahalanobisDistance(double[] x)
	{
		int k						= active.size();
		ContStats distanceSeries	= new ContStats(weighted);
		
		// Case for scalar or diagonal bandwidths
		if (!Double.isNaN(scalarBandwidth) || diagBandwidth != null)
		{
			for (int s = 0; s < samples.size(); s++)
			{
				double sum			= 0.0;
				ContMultiSample sample	= samples.get(s);
				ArrayList<Double> values = sample.getValues();
				for (int d = 0; d < k; d++)
				{
					int index		= active.get(d);
					double diff		= values.get(index) - x[index];
					double var = diagBandwidth == null ? scalarBandwidth : diagBandwidth[index];
					sum				+= diff*diff/var;
				}
				double weight		= weighted ? sample.getWeight() : 1.0;
				distanceSeries.addValue(sum, weight);
			}
			return distanceSeries.getMean();
		}
		
		// Prepare matrices
		double[][] matrix			= invertible.getC();
		double[][] lower			= MatUtil.cholesky(matrix);
		
		// Compute distances
		for (int s = 0; s < samples.size(); s++)
		{
			ContMultiSample sample	= samples.get(s);
			double[] dev			= new double[k];
			for (int i = 0; i < k ; i++)
			{
				int index			= active.get(i);
				dev[i]				= x[index] - sample.getValues().get(index);
			}
			double[] y				= MatUtil.solveLower(lower, dev);
			double mahalanobis		= 0.0;
			for (int i = 0; i < y.length; i++)
				mahalanobis			+= y[i]*y[i];
			double weight			= weighted ? sample.getWeight() : 1.0;
			distanceSeries.addValue(mahalanobis, weight);
		}
		return distanceSeries.getMean();
	}

	/**
	 * @param x	The point to compute the distance from
	 * @return	The Mahalanobis distance between the provided point and each of the samples that
	 * 			define the distribution
	 */
	public double[] getMahalanobisDistanceToSamples(double[] x)
	{
		int k						= active.size();
		double[] distance			= new double[samples.size()];
		
		// Case for scalar or diagonal bandwidths
		if (!Double.isNaN(scalarBandwidth) || diagBandwidth != null)
		{
			for (int s = 0; s < samples.size(); s++)
			{
				double sum			= 0.0;
				ArrayList<Double> values = samples.get(s).getValues();
				for (int d = 0; d < k; d++)
				{
					int index		= active.get(d);
					double diff		= values.get(index) - x[index];
					double var = diagBandwidth == null ? scalarBandwidth : diagBandwidth[index];
					sum				+= diff*diff/var;
				}
				distance[s]			= sum;
			}
			return distance;
		}
		
		// Prepare matrices
		double[][] matrix			= invertible.getC();
		double[][] lower			= MatUtil.cholesky(matrix);
		
		// Compute distances
		for (int s = 0; s < samples.size(); s++)
		{
			ContMultiSample sample	= samples.get(s);
			double[] dev			= new double[k];
			for (int i = 0; i < k ; i++)
			{
				int index			= active.get(i);
				dev[i]				= x[index] - sample.getValues().get(index);
			}
			double[] y				= MatUtil.solveLower(lower, dev);
			double mahalanobis		= 0.0;
			for (int i = 0; i < y.length; i++)
				mahalanobis			+= y[i]*y[i];
			distance[s]				= mahalanobis;
		}
		return distance;
	}
	
	@Override
	public double getMahalanobisForce(double[] x)
	{
		int k						= active.size();
		double[][] matrix			= invertible.getC();
		double[][] lower			= MatUtil.cholesky(matrix);
		double sum					= 0;
		double weightSum			= 0;
		for (ContMultiSample sample : samples)
		{
			double[] dev			= new double[k];
			for (int i = 0; i < k ; i++)
			{
				int index			= active.get(i);
				dev[i]				= x[index] - sample.getValues().get(index);
			}
			double[] y				= MatUtil.solveLower(lower, dev);
			double sqrdMahalanobis	= 0.0;
				for (int i = 0; i < y.length; i++)
					sqrdMahalanobis	+= y[i]*y[i];
			double weight			= sample.getWeight();
			double inverseSqrd		= 1/(Math.abs(sqrdMahalanobis));
			sum						+= weighted ? inverseSqrd/weight : inverseSqrd;
			weightSum				+= weighted ? weight : 1.0;
		}
		return weightSum/sum;
	}
	
	/**
	 * Computes the mean Mahalanobis force at the provided point. The force is defined here as the
	 * mean weighted inverse squared Mahalanobis distance between the point and the samples.
	 * @param x	The point to evaluate the force at
	 * @return	The mean Mahalanobis force at the provided point
	 */
	public double getMeanMahalanobisForce(double[] x)
	{
		int k						= active.size();
		double[][] matrix			= invertible.getC();
		double[][] lower			= MatUtil.cholesky(matrix);
		double sum					= 0;
		double weightSum			= 0;
		for (ContMultiSample sample : samples)
		{
			double[] dev			= new double[k];
			for (int i = 0; i < k ; i++)
			{
				int index			= active.get(i);
				dev[i]				= x[index] - sample.getValues().get(index);
			}
			double[] y				= MatUtil.solveLower(lower, dev);
			double sqrdMahalanobis	= 0.0;
				for (int i = 0; i < y.length; i++)
					sqrdMahalanobis	+= y[i]*y[i];
			double weight			= sample.getWeight();
			double inverseSqrd		= 1/(Math.abs(sqrdMahalanobis));
			sum						+= weighted ? weight*inverseSqrd : inverseSqrd;
			weightSum				+= weighted ? weight : 1.0;
		}
		return sum/weightSum;
	}

	/**
	 * Computes the mean Mahalanobis force at the provided point. The force is defined here as the
	 * mean weighted inverse squared Mahalanobis distance between the point and the samples.
	 * @param x	The point to evaluate the force at
	 * @return	The mean Mahalanobis force at the provided point
	 */
	public double getMeanMahalanobisForce(ArrayList<Double> x)
	{
		return getMeanMahalanobisForce(Utilities.toArray(x));
	}

	/**
	 * Marginalizes the distribution of one of the dimensions. That is, creates a reduced 
	 * distribution of only one of the original dimensions.
	 * <p>Time complexity: <i>O(number of samples)</i>
	 * <br>Space complexity: <i>O(number of samples)</i>
	 * @param dimension	The index of the dimension to marginalize
	 * @return			The marginalized kernel density distribution
	 */
	public KernelDensity marginalize(int dimension)
	{
		if (dimension >= statistics.size() || samples.size() == 0)
			return null;
		
		KernelDensity marginal	= new KernelDensity();
		marginal.setWeighted(weighted);
		for (ContMultiSample sample : samples)
			marginal.addSample(sample.getValues().get(dimension), sample.getWeight());
		
		double variance			= Double.NaN;
		if (!Double.isNaN(scalarBandwidth))
			variance			= scalarBandwidth;
		else if (diagBandwidth != null)
			variance			= diagBandwidth[dimension];
		else
		{
			int index			= active.indexOf(dimension);
			if (index > -1)
				variance		= invertible.getC()[index][index];
			else
			{
				index			= zeroVar.indexOf(dimension);
				if (index > -1)
					variance	= 0.0;
				else
					for (PointID coeff : regressionCoeff.get(dimension))
						if (coeff.getX() == -2)
						{
							variance = coeff.getY();
							break;
						}
			}
		}
		marginal.setBandwidth(Math.sqrt(variance));
		
		return marginal;
	}
	
	/**
	 * Marginalizes the distribution of a list of the dimensions. That is, creates a list of 
	 * reduced distribution of each of the selected dimensions.
	 * <p>Time complexity: <i>O(number of samples*number of dimensions)</i>
	 * <br>Space complexity: <i>O(number of samples*number of dimensions)</i>
	 * @param toActivate	A list containing the indices of the dimensions to marginalize
	 * @return				The list of marginalized kernel density distributions
	 */
	public ArrayList<KernelDensity> marginalizeMultiple(ArrayList<Integer> toActivate)
	{
		if (samples.size() == 0)
			return null;
		
		// Create distributions and assign bandwidth
		ArrayList<KernelDensity> marginals	= new ArrayList<>(toActivate.size());
		for (int d = 0; d < toActivate.size(); d++)
		{
			Integer dimension				= toActivate.get(d);
			KernelDensity marginal			= null;
			if (dimension >= 0 && dimension < statistics.size())
			{
				marginal					= new KernelDensity();
				marginal.setWeighted(weighted);
				double variance				= Double.NaN;
				if (!Double.isNaN(scalarBandwidth))
					variance				= scalarBandwidth;
				else if (diagBandwidth != null)
					variance				= diagBandwidth[dimension];
				else
				{
					int index				= active.indexOf(dimension);
					if (index > -1)
						variance			= invertible.getC()[index][index];
					else
					{
						index				= zeroVar.indexOf(dimension);
						if (index > -1)
							variance		= 0.0;
						else
							for (PointID coeff : regressionCoeff.get(dimension))
								if (coeff.getX() == -2)
								{
									variance = coeff.getY();
									break;
								}
					}
				}
				marginal.setBandwidth(Math.sqrt(variance));
			}
			marginals.add(marginal);
		}
		
		// Add samples
		for (ContMultiSample sample : samples)
		{
			ArrayList<Double> values		= sample.getValues();
			double weight					= sample.getWeight();
			for (int m = 0; m < marginals.size(); m++)
			{
				KernelDensity marginal		= marginals.get(m);
				int index					= toActivate.get(m);
				if (marginal != null)
					marginal.addSample(values.get(index), weight);
			}
		}
		return marginals;
	}

	/**
	 * Marginalizes the distribution of a set of variables. That is, creates a reduced distribution 
	 * with only a set of the original dimensions.
	 * <p>Time complexity: <i>O(number of samples * number of dimensions to marginalize)</i>
	 * <br>Space complexity: <i>O(number of samples * number of dimensions to marginalize)</i>
	 * @param toActivate A list containing the indices of the dimensions to marginalize
	 * @return The marginalized kernel density distribution. <code>null</code> if the activation
	 * parameter contained no dimensions to marginalize.
	 */
	public MultiVarKernelDensity marginalize(ArrayList<Integer> toActivate)
	{
		return marginalize(toActivate, true);
	}
	
	/**
	 * Marginalizes the distribution of a set of variables. That is, creates a reduced distribution 
	 * with only a set of the original dimensions.
	 * <p>Time complexity: <i>O(number of samples * number of dimensions to marginalize)</i>
	 * <br>Space complexity: <i>O(number of samples * number of dimensions to marginalize)</i>
	 * @param toActivate A list containing the indices of the dimensions to marginalize
	 * @param keepBandwidth	True if the bandwidth information should be passed to the marginalized
	 * 						distribution. False otherwise (the bandwidth then needs to be 
	 * 						determined afterwards).
	 * @return The marginalized kernel density distribution. <code>null</code> if the activation
	 * parameter contained no dimensions to marginalize.
	 */
	public MultiVarKernelDensity marginalize(ArrayList<Integer> toActivate, boolean keepBandwidth)
	{
		MultiVarKernelDensity marginal		= new MultiVarKernelDensity();
		marginal.setWeighted(weighted);
		Collections.sort(toActivate);
		
		// Verify dimensions to marginalize
		if (toActivate.isEmpty())
			return null;
		
		// Create samples
		for (ContMultiSample sample : samples)
		{
			ArrayList<Double> values = new ArrayList<Double>();
			for (Integer activate : toActivate)
				values.add(sample.getValues().get(activate));
			ContMultiSample marginalSample = new Sample(sample.getWeight(), values);
			marginal.addSample(marginalSample);
		}
		
		if (!keepBandwidth)
			return marginal;
		
		// Determine bandwidth
		int dimCount						= toActivate.size();					
		if (invertible != null)
		{
			int[] dims						= new int[dimCount];
			for (int i = 0; i < dimCount; i++)
				dims[i]						= toActivate.get(i);
			RealMatrix bandwidthMat			= new Array2DRowRealMatrix(getBandwidth());
			RealMatrix marginalBwMat		= bandwidthMat.getSubMatrix(dims, dims);
			marginal.setBandwidth(marginalBwMat.getData());
		}
		else if (diagBandwidth != null)
		{
			ArrayList<Double> margBW		= new ArrayList<>();
			for (Integer index : toActivate)
				margBW.add(diagBandwidth[index]);
			marginal.setBandwidth(margBW);
		}
		else
			marginal.setBandwidth(scalarBandwidth);
		
		return marginal;
	}

	/**
	 * Computes a conditional probability distribution by creating a copy of the original 
	 * distribution. The weights of the samples on the conditioned distribution are adjusted so 
	 * that they are conditioned on the provided particle or value vector. That is, the original 
	 * weights are multiplied by the corresponding likelihood of the particle given the kernel on 
	 * each sample.
	 * @param particle The particle to condition the copied distribution on by adjusting the weight
	 * of its samples. Values with the constant {@link java.lang.Double#NaN} are not used for the
	 * conditioning. That is, a marginalized version of the distribution is used to compute the 
	 * likelihood values.
	 * @return A copy of the distribution conditioned on the provided particle
	 */
	public MultiVarKernelDensity conditional(ArrayList<Double> particle)
	{
		MultiVarKernelDensity conditional 	= new MultiVarKernelDensity();
		conditional.setWeighted(true);
		
		// Retrieve dimensions to marginalize
		ArrayList<Integer> toMarginalize	= new ArrayList<>();
		ArrayList<Double> marginalPart		= new ArrayList<Double>();
		int lastDimension 					= Math.min(statistics.size(), particle.size());
		for (int j = 0; j < lastDimension; j++)
		{
			double value					= particle.get(j);
			if (!Double.isNaN(value))
			{
				toMarginalize.add(j);
				marginalPart.add(value);
			}
		}
		double[] part						= new double[marginalPart.size()];
		for (int j = 0; j < marginalPart.size(); j++)
			part[j]							= marginalPart.get(j);
		
		// Compute marginal and initialize conditional
		MultiVarKernelDensity marginal		= marginalize(toMarginalize);
		if (marginal == null)
			return null;
		
		// Compute conditional weights
		double[][] bandwidth				= marginal.getBandwidth();
		for (int i = 0; i < samples.size(); i++)
		{
			ContMultiSample marginalSample	= marginal.getSamples().get(i);
			ArrayList<Double> values		= marginalSample.getValues();
			double[] mean					= new double[values.size()];
			for (int j = 0; j < values.size(); j++)
				mean[j]						= values.get(j);
			MultiVarNormal kernel			= new MultiVarNormal(mean, bandwidth);
			double likelihood				= kernel.getpdf(part);
			ContMultiSample condSample		= samples.get(i).copy();
			condSample.setWeight(condSample.getWeight()*likelihood);
			conditional.addSample(condSample);
		}
		
		// Check weight sum
		if (conditional.getWeightSum() == 0.0)
		{
			for (ContMultiSample sample : conditional.getSamples())
				sample.setWeight(Sample.DEF_WEIGHT);
			conditional.computeStatistics();
		}
		
		// TODO Scale covariance or recompute bandwidth matrix
		
		// Set bandwidth
		if (!Double.isNaN(scalarBandwidth))
			conditional.setBandwidth(scalarBandwidth);
		else if (diagBandwidth != null)
			conditional.setBandwidth(diagBandwidth);
		else
			conditional.setBandwidth(getBandwidth());
		
		return conditional;
	}
	
	@Override
	public double[] sample()
	{
		return Utilities.toArray(sampleOb().getValues());
	}
	
	/**
	 * Generates a random sample from this distribution. Returns <code>null</code> if the 
	 * distribution has not been correctly defined.
	 * @return A random sample from this distribution
	 */
	public Sample sampleOb()
	{
		if (samples.size() == 0)
			return null;
		
		int dimensions					= statistics.size();
		ArrayList<Double> vector		= new ArrayList<>(dimensions);
		for (int d = 0; d < dimensions; d++)
			vector.add(Double.NaN);
		
		// Add zero-variance variables
		if (zeroVar != null)
			for (Integer index : zeroVar)
				vector.set(index, statistics.get(index).getMean());
		
		// Select sample
		double weightSum				= statistics.get(0).getWeightSum();
		double root						= Math.random()*weightSum;
		double sum 						= 0;
		int index						= -1;
		ContMultiSample sample			= null;
		while (root >= sum || sample == null)
		{
			index++;
			sample						= (ContMultiSample) samples.get(index);
			sum							+= sample.getWeight();
		}
		ArrayList<Double> mean			= sample.getValues();
		
		// Generate random values
		if (!Double.isNaN(scalarBandwidth))
		{
			double stDev				= Math.sqrt(scalarBandwidth);
			for (int d = 0; d < dimensions; d++)
				vector.set(d, Normal.sample(mean.get(d), stDev));
		}
		else if (diagBandwidth != null)
			for (Integer d : active)
			{
				double stDev			= Math.sqrt(diagBandwidth[d]);
				vector.set(d, Normal.sample(mean.get(d), stDev));
			}
		else
		{
			int activeCount				= active.size();
			MultiVarNormal kernel		= invertible;
			double[] kernelMean			= new double[active.size()];
			for (int d = 0; d < activeCount; d++)
				kernelMean[d]			= mean.get(active.get(d));
			kernel.setMean(kernelMean);
			double[] values				= kernel.sample();
			for (int d = 0; d < activeCount; d++)
				vector.set(active.get(d), values[d]);
		}
		
		// Add linearly-dependent variables
		if (linearlyDependent != null)
			for (Integer index2 : linearlyDependent)
			{
				ArrayList<PointID> reg	= regressionCoeff.get(index2);
				double value			= 0.0;
				double var				= 0.0;
				for (int t = 0; t < reg.size(); t++)
				{
					PointID point		= reg.get(t);
					index				= point.x;
					if (index == -2)
						var				= point.getY();
					else
						value			+= index == -1 ? point.y : vector.get(index)*point.y;
				}
				if (var > 0.0)
					value				= Normal.sample(value, Math.sqrt(var));
				vector.set(index2, value);
			}

		return new Sample(1.0, vector);
	}
	
	@Override
	public double[][] sampleMultiple(int count)
	{
		ArrayList<Sample> sampArr		= sampleMultipleOb(count);
		int dimensions					= statistics.size();
		double[][] samples				= new double[sampArr.size()][dimensions];
		for 	(int s = 0; s < sampArr.size();	s++)
		{
			ArrayList<Double> values	= sampArr.get(s).getValues();
			for	(int d = 0; d < dimensions;		d++)
				samples[s][d]			= values.get(d);
		}
		return samples;
	}

	/**
	 * Generates a defined number of random samples from this distribution. Returns 
	 * <code>null</code> if the distribution has not been correctly defined.
	 * @param count The number of random samples to generate
	 * @return A list of random samples from this distribution
	 */
	public ArrayList<Sample> sampleMultipleOb(int count)
	{
		if (count == 0)
			return new ArrayList<>();
		
		if (samples.size() == 0)
			return null;
		
		ArrayList<Sample> result	= new ArrayList<>();
		for (int s = 0; s < count; s++)
			result.add(sampleOb());
		return result;
	}
	
}