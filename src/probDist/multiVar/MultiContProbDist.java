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

import probDist.ContProbDist;
import utilities.stat.ContSeries;

/**
 * This class represents multivariate probability distributions for continuous random variables
 * @author Felipe Hernández
 */
public abstract class MultiContProbDist
{

	// --------------------------------------------------------------------------------------------
	// Constants
	// --------------------------------------------------------------------------------------------
	
	/**
	 * Distribution type identifier: normal or Gaussian
	 */
	public final static int NORMAL = 1;
	
	/**
	 * Distribution type identifier: lightweight Gaussian graphical model
	 */
	public final static int GGMLITE = 2;
	
	/**
	 * Distribution type identifier: kernel density
	 */
	public final static int KERNEL = 3;
	
	/**
	 * Distribution type identifier: lightweight kernel density Gaussian graphical model
	 */
	public final static int KERNEL_GGMLITE = 4;
	
	/**
	 * Distribution type identifier: mixture
	 */
	public final static int MIXTURE = 5;
	
	/**
	 * The base to compute the default number of samples to estimate the error between two 
	 * distributions using the default method. The number of samples is computed as: 
	 * <i>num_samp = base^(dim^exp)</i>.
	 */
	public final static double ERROR_SAMPLE_BASE = 10;
	
	/**
	 * The exponent of the number of dimensions to compute the default number of samples to 
	 * estimate the error between two distributions using the default method. The number of samples 
	 * is computed as: <i>num_samp = base^(dim^exp)</i>.
	 */
	public final static double ERROR_SAMPLE_DIM_EXP = 0.4;
	
	/**
	 * The maximum number of samples to estimate the error between two distributions using the 
	 * default method
	 */
	public final static double ERROR_SAMPLE_MAX = 5000;
	
	// --------------------------------------------------------------------------------------------
	// Attributes
	// --------------------------------------------------------------------------------------------
	
	/**
	 * Identifier of the type of distribution
	 */
	protected int type;
	
	/**
	 * The partial probability when used as a component in a mixture distribution
	 */
	protected double prob;
	
	// --------------------------------------------------------------------------------------------
	// Methods
	// --------------------------------------------------------------------------------------------
	
	/**
	 * @return {@link #type}
	 */
	public int getType()
	{
		return type;
	}
	
	/**
	 * @return {@link #prob}
	 */
	public double getProb() 
	{
		return prob;
	}

	/**
	 * @param prob {@link #prob}
	 */
	public void setProb(double prob)
	{
		this.prob = prob < 0 ? 0 : prob;
	}
	
	/**
	 * Computes the root mean square error (RMSE) between the probability density function of this
	 * distribution and another one. A default number of test points to evaluate the density, 
	 * which depends on the number of dimensions, is used. The test points are sampled randomly 
	 * from this distribution.
	 * @param other	The distribution to compare with (must have the same dimensions)
	 * @return		The root mean square error (RMSE) between the probability density function of 
	 * this distribution and another one.
	 */
	public double computeRMSE(MultiContProbDist other)
	{
		return computeRMSE(other, getErrorSamples());
	}
	
	/**
	 * Computes the root mean square error (RMSE) between the probability density function of this
	 * distribution and another one. The density is compared at a specified number of test points. 
	 * The test points are sampled randomly from this distribution.
	 * @param other			The distribution to compare with (must have the same dimensions)
	 * @param sampleCount	The number of test points to compare the distributions
	 * @return				The root mean square error (RMSE) between the probability density 
	 * 						function of this distribution and another one.
	 */
	public double computeRMSE(MultiContProbDist other, int sampleCount)
	{
		double[][] samples			= sampleMultiple(sampleCount);
		double sum					= 0;			
		for (int s = 0; s < sampleCount; s++)
		{
			double[] sample			= samples[s];
			double pdfThis			= getpdf(sample);
			double pdfOther			= other.getpdf(sample);
			double error			= pdfThis - pdfOther;
			sum						+= error*error;
		}
		return Math.sqrt(sum/sampleCount);
	}
	
	/**
	 * Computes the mean absolute relative error (MARE) between the probability density function of 
	 * this distribution and another one. The density is compared at a specified number of test 
	 * points. The test points are sampled randomly from this distribution.
	 * @param other	The distribution to compare with (must have the same dimensions)
	 * @return		The mean absolute relative error (MARE) between the probability density 
	 * 				function of this distribution and another one.
	 */
	public double computeMARE(MultiContProbDist other)
	{
		return computeMARE(other, getErrorSamples());
	}
	
	/**
	 * Computes the mean absolute relative error (MARE) between the probability density function of 
	 * this distribution and another one. The density is compared at a specified number of test 
	 * points. The test points are sampled randomly from this distribution.
	 * @param other			The distribution to compare with (must have the same dimensions)
	 * @param sampleCount	The number of test points to compare the distributions
	 * @return				The mean absolute relative error (MARE) between the probability density 
	 * 						function of this distribution and another one.
	 */
	public double computeMARE(MultiContProbDist other, int sampleCount)
	{
		double[][] samples			= sampleMultiple(sampleCount);
		ContSeries series			= new ContSeries(false);
		for (int s = 0; s < sampleCount; s++)
		{
			double[] sample			= samples[s];
			double pdfThis			= getpdf(sample);
			double pdfOther			= other.getpdf(sample);
			series.addValue(Math.abs((pdfThis - pdfOther)/pdfThis));
		}
		return series.getMean();
	}
	
	/**
	 * @return The number of samples to estimate the error between two distributions. The number of 
	 * samples is computed as: <i>num_samp = base^(dim^exp)</i>.
	 */
	private int getErrorSamples()
	{
		double base	= ERROR_SAMPLE_BASE;
		double exp	= ERROR_SAMPLE_DIM_EXP;
		double dim	= getDimensionality();
		return (int) Math.min(ERROR_SAMPLE_MAX, Math.pow(base, Math.pow(dim, exp)));
	}
	
	/**
	 * Returns the String identifier of the type of the distribution. The different types are 
	 * defined as class constants
	 * @return The String identifier of the type of distribution
	 */
	public abstract String getTypeString();
	
	/**
	 * @return The number of dimensions or random variables (<i>k</i>)
	 */
	public abstract int getDimensionality();
	
	/**
	 * @return A <i>k</i>-sized vector with the mean value of each dimension or random variable
	 */
	public abstract double[] getMean();
	
	/**
	 * @return A <i>k</i>-sized vector with the marginal variance values of each dimension or 
	 * random variable
	 */
	public abstract double[] getVariance();
	
	/**
	 * @return The <i>k</i> x <i>k</i> covariance matrix of the distribution, with the marginal 
	 * variance values in the diagonal, and the covariance terms elsewhere.
	 */
	public abstract double[][] getCovariance();
	
	/**
	 * Computes the value of the probability density function at the provided point. Returns 
	 * {@link java.lang.Double#NaN} if the distribution has not been correctly defined.
	 * @param x	The point at which to evaluate the probability density
	 * @return	The value of the probability density function at the provided point
	 */
	public abstract double getpdf(ArrayList<Double> x);
	
	/**
	 * Computes the value of the probability density function at the provided point. Returns 
	 * {@link java.lang.Double#NaN} if the distribution has not been correctly defined.
	 * @param x	The point at which to evaluate the probability density
	 * @return	The value of the probability density function at the provided point
	 */
	public abstract double getpdf(double[] x);
	
	/**
	 * Computes the value of the natural logarithm of the probability density function at the 
	 * provided point. Returns {@link java.lang.Double#NaN} if the distribution has not been 
	 * correctly defined.
	 * @param x The point at which to evaluate the probability density
	 * @return The value of the natural logarithm of the probability density function at the 
	 * provided point
	 */
	public abstract double getLogpdf(ArrayList<Double> x);
	
	/**
	 * Computes the value of the natural logarithm of the probability density function at the 
	 * provided point. Returns {@link java.lang.Double#NaN} if the distribution has not been 
	 * correctly defined.
	 * @param x The point at which to evaluate the probability density
	 * @return The value of the natural logarithm of the probability density function at the 
	 * provided point
	 */
	public abstract double getLogpdf(double[] x);
	
	/**
	 * Marginalizes the distribution of one of the dimensions. That is, creates a reduced 
	 * distribution of only one of the original dimensions.
	 * @param dimension	The index of the dimension to marginalize
	 * @return			The marginalized distribution
	 */
	public abstract ContProbDist marginalize(int dimension);
	
	/**
	 * Marginalizes the distribution of a set of variables. That is, creates a reduced distribution 
	 * with only a set of the original dimensions.
	 * @param toActivate	A set containing the indices of the dimensions to marginalize
	 * @return				The marginalized distribution. <code>null</code> if the activation 
	 * parameter contained no dimensions to marginalize.
	 */
	public abstract MultiContProbDist marginalize(ArrayList<Integer> toActivate);
	
	/**
	 * Computes a conditional probability distribution given a set of values for some of the
	 * dimensions or random variables.
	 * @param x	The values to condition the resulting distribution on. The vector should have size
	 * <i>k</i> (the total number of dimensions of the distribution), but values with the constant 
	 * {@link java.lang.Double#NaN} are not used for the conditioning, and will correspond to the 
	 * ones featured in the conditioned (posterior) distribution.
	 * @return A conditional probability distribution
	 */
	public abstract MultiContProbDist conditional(ArrayList<Double> x);
	
	/**
	 * @return A random vector from the multivariate normal distribution
	 */
	public abstract double[] sample();
	
	/**
	 * Generates a defined number of random samples from this distribution. Returns 
	 * <code>null</code> if the distribution has not been correctly defined.
	 * @param count	The number of random samples to generate
	 * @return		An matrix where each row represents a generated random sample and each column
	 * 				one of the dimensions or random variables
	 */
	public abstract double[][] sampleMultiple(int count);
	
}
