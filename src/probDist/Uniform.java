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

package probDist;

import utilities.Utilities;
import utilities.stat.ContSeries;

/**
 * This class represents uniform probability distributions and defines static methods for making 
 * computations with uniform probability distributions. Available methods include: <ul>
 * <li>Compute the probability of a range in a uniform distribution
 * <li>Compute the expectation of a range of a uniform distribution
 * <li>Compute the CDF and inverse CDF functions for a uniform distribution
 * <li>Generate random numbers with a uniform distribution </ul>
 * @author Felipe Hernández
 */
public class Uniform extends ContProbDist
{
	
	// --------------------------------------------------------------------------------------------
	// Constants
	// --------------------------------------------------------------------------------------------
	
	/**
	 * Uniform distribution type String identifier
	 */
	public final static String ID = "Uniform";
	
	/**
	 * Uniform distribution type short String identifier
	 */
	public final static String SHORT_ID = "U";
	
	// --------------------------------------------------------------------------------------------
	// Attributes
	// --------------------------------------------------------------------------------------------
	
	/**
	 * The minimum value of the distribution
	 */
	private double min;
	
	/**
	 * The maximum value of the distribution
	 */
	private double max;
	
	// --------------------------------------------------------------------------------------------
	// Constructors
	// --------------------------------------------------------------------------------------------
	
	/**
	 * Creates a uniform probability distribution
	 * @param min The minimum value of the distribution
	 * @param max The maximum value of the distribution
	 */
	public Uniform(double min, double max)
	{
		type = ContProbDist.UNIFORM;
		min = min <= max ? min : max;
		max = max >= min ? max : min;
		this.min = min;
		this.max = max;
	}
	
	/**
	 * Creates a uniform probability distribution from a set of observed data. The distribution's
	 * parameters are computed by using the mean and standard deviation of the data
	 * @param values The data points to build the distribution from
	 */
	public Uniform(ContSeries values)
	{
		type = ContProbDist.UNIFORM;
		double mean = values.getMean();
		double stdDev = values.getStDev();
		double extent = stdDev*Math.sqrt(12);
		min = mean - extent/2;
		max = mean + extent/2;
	}
	
	// --------------------------------------------------------------------------------------------
	// Non-static methods
	// --------------------------------------------------------------------------------------------
	
	/**
	 * @return The minimum value of the distribution
	 */
	public double getMin() 
	{
		return min;
	}

	/**
	 * @param min The minimum value of the distribution
	 */
	public void setMin(double min) 
	{
		this.min = min;
	}

	/**
	 * @return The maximum value of the distribution
	 */
	public double getMax() 
	{
		return max;
	}

	/**
	 * @param max The maximum value of the distribution
	 */
	public void setMax(double max) 
	{
		this.max = max;
	}

	@Override
	public double getMean()
	{
		return (max + min)/2;
	}
	
	@Override
	public double getStDev()
	{
		return (max - min)/Math.sqrt(12);
	}
	
	@Override
	public double getSkewness()
	{
		return 0.0;
	}
	
	@Override
	public double getVar()
	{
		double range = max - min;
		return range*range/12;
	}
	
	@Override
	public String getTypeString() 
	{
		return ID;
	}

	@Override
	public String toString() 
	{
		return SHORT_ID + "(" + min + ", " + max + ")";
	}

	@Override
	public String toString(int decimalPlaces) 
	{
		double roundedMin = Utilities.round(min, decimalPlaces);
		double roundedMax = Utilities.round(max, decimalPlaces);
		return SHORT_ID + "(" + roundedMin + ", " + roundedMax + ")";
	}
	
	@Override
	public ContProbDist truncate(double min, double max)
	{
		return new Uniform(min, max);
	}
	
	@Override
	public double getpdf(double x) 
	{
		return computepdf(min, max, x);
	}

	@Override
	public double getCDF(double x) 
	{
		return computeCDF(min, max, x);
	}

	@Override
	public double getInvCDF(double p) 
	{
		return computeInvCDF(min, max, p);
	}

	@Override
	public double sample() 
	{
		return sample(min, max);
	}
	
	@Override
	public void shift(double newMean)
	{
		double shift	= newMean - getMean();
		setMin(min + shift);
		setMax(max + shift);
	}

	@Override
	public void scale(double newStDev)
	{
		double mean		= getMean();
		double delta	= newStDev*Math.sqrt(12);
		setMin(mean - delta/2);
		setMax(mean + delta/2);
	}
	
	// --------------------------------------------------------------------------------------------
	// Static methods
	// --------------------------------------------------------------------------------------------

	/**
	 * Computes the mean of a random variable with a given uniform distribution
	 * @param min The minimum value of the distribution
	 * @param max The maximum value of the distribution
	 * @return The mean of a random variable with a given uniform distribution
	 */
	public static double computeMean(double min, double max)
	{
		return (max + min)/2;
	}
	
	/**
	 * Computes the standard deviation of a random variable with a given uniform distribution
	 * @param min The minimum value of the distribution
	 * @param max The maximum value of the distribution
	 * @return The standard deviation of a random variable with a given uniform distribution
	 */
	public static double computeStDev(double min, double max)
	{
		return (max - min)/Math.sqrt(12);
	}
	
	/**
	 * Computes the value of the probability density function of a uniform probability 
	 * distribution. Returns {@link java.lang.Double#NaN} if <code>min</code> is larger than 
	 * <code>max</code>.
	 * @param min The minimum value of the distribution
	 * @param max The maximum value of the distribution
	 * @param x The value at which to compute the probability density
	 * @return The value of the probability density function of a uniform probability 
	 * distribution
	 */
	public static double computepdf(double min, double max, double x)
	{
		if (min >= max)
			return Double.NaN;
		else if (x < min || x > max)
			return 0;
		else
			return 1/(max - min);
	}
	
	/**
	 * Computes the value of the cumulative distribution function of a uniform probability 
	 * distribution. Returns Double.NaN if the value of max is not greater than the values of min.
	 * @param min The minimum value of the distribution
	 * @param max The maximum value of the distribution
	 * @param z The value or quantile
	 * @return The value of the cumulative distribution function of a uniform probability 
	 * distribution
	 */
	public static double computeCDF(double min, double max, double z)
	{
		if(min >= max)
			return Double.NaN;
		else if(z < min)
			return 0;
		else if(z > max)
			return 1;
		else
			return (z - min)/(max - min);
	}
	
	/**
	 * Computes the value of a random variable with a uniform distribution that has the cumulative 
	 * distribution function value that enters as a parameter. That is, the inverse cumulative 
	 * distribution function value. Returns Double.NaN if the values of max is not greater than the 
	 * values of min.
	 * @param min The minimum value of the distribution
	 * @param max The maximum value of the distribution
	 * @param p The value of the cumulative distribution function
	 * @return The value of a random variable with a uniform distribution that has the cumulative 
	 * distribution function value that enters as a parameter
	 */
	public static double computeInvCDF(double min, double max, double p) 
	{
		if(p < 0 || p > 1)
			return Double.NaN;
		if(min >= max)
			return Double.NaN;
		else
			return min + p*(max - min);
	}
	
	/**
	 * Computes the root mean square error from comparing the accumulated weight function of a 
	 * series of data points to the cumulative probability density function of a uniform 
	 * distribution. This indicators helps measure how well the uniform distribution fits the data.
	 * @param min The minimum value of the distribution
	 * @param max The maximum value of the distribution
	 * @param values The data points to be compared
	 * @return The root mean square error from comparing the accumulated weight function of a 
	 * series of data points to the cumulative probability density function of a uniform 
	 * distribution
	 */
	public static double computeRMSE(double min, double max, ContSeries values)
	{
		Uniform uniform = new Uniform(min, max);
		return uniform.computeRMSE(values);
	}
	
	/**
	 * Generates a uniformly distributed random number
	 * @param min The minimum value of the distribution
	 * @param max The maximum value of the distribution
	 * @return A uniformly distributed random number
	 */
	public static double sample(double min, double max)
	{
		return min + Math.random()*(max - min);
	}
	
}