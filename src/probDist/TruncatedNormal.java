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

/**
 * This class represents truncated normal probability distributions; that is, normal distributions
 * constrained between a given lower and and upper limit. It also defines static methods for making 
 * computations with truncated normal probability distributions. Available methods include: <ul>
 * <li>Compute the probability of a range in a truncated normal probability distribution
 * <li>Compute the CDF and inverse CDF functions for a truncated normal probability distribution
 * <li>Generate normally distributed random numbers within a range </ul>
 * @author Felipe Hernández
 */
public class TruncatedNormal extends ContProbDist
{

	// --------------------------------------------------------------------------------------------
	// Constants
	// --------------------------------------------------------------------------------------------
	
	/**
	 * Normal distribution type String identifier
	 */
	public final static String ID = "Truncated normal";
	
	/**
	 * Normal distribution type short String identifier
	 */
	public final static String SHORT_ID = "TN";
	
	/**
	 * Inverse of the square root of two times pi
	 */
	private final static double DOUBLE_PI_SQRT_INV = 0.398942280401433;
	
	// --------------------------------------------------------------------------------------------
	// Attributes
	// --------------------------------------------------------------------------------------------
	
	/**
	 * Corresponds to the mean parameter of a non-truncated normal distribution
	 */
	private double location;
	
	/**
	 * Corresponds to the standard deviation parameter of a non-truncated normal distribution
	 */
	private double scale;
	
	/**
	 * The lower truncation bound (exclusive); lower limit value that the distribution can take
	 */
	private double min;
	
	/**
	 * The upper truncation bound (exclusive); upper limit value that the distribution can take
	 */
	private double max;
	
	// --------------------------------------------------------------------------------------------
	// Constructors
	// --------------------------------------------------------------------------------------------
	
	/**
	 * @param location	{@link #location}
	 * @param scale		{@link #scale}
	 * @param min		{@link #min}
	 * @param max		{@link #max}
	 */
	public TruncatedNormal(double location, double scale, double min, double max)
	{
		type			= ContProbDist.TRUNC_NORMAL;
		this.location	= location;
		this.scale		= scale < 0.0 ? 0.0 : scale;
		this.min		= min;
		this.max		= max;
	}
	
	// --------------------------------------------------------------------------------------------
	// Non-static methods
	// --------------------------------------------------------------------------------------------
	
	/**
	 * @return {@link #location}
	 */
	public double getLocation()
	{
		return location;
	}

	/**
	 * @param location {@link #location}
	 */
	public void setLocation(double location)
	{
		this.location = location;
	}

	/**
	 * @return {@link #scale}
	 */
	public double getScale()
	{
		return scale;
	}

	/**
	 * @param scale {@link #scale}
	 */
	public void setScale(double scale)
	{
		this.scale = scale < 0.0 ? 0.0 : scale;
	}

	/**
	 * @return {@link #min}
	 */
	public double getMin()
	{
		return min;
	}

	/**
	 * @param min {@link #min}
	 */
	public void setMin(double min)
	{
		this.min = min;
	}

	/**
	 * @return {@link #max}
	 */
	public double getMax()
	{
		return max;
	}

	/**
	 * @param max {@link #max}
	 */
	public void setMax(double max)
	{
		this.max = max;
	}

	@Override
	public String getTypeString()
	{
		return ID;
	}

	@Override
	public String toString()
	{
		return SHORT_ID + "(" + location + ", " + scale + ", " + min + " < x < " + max + ")";
	}

	@Override
	public String toString(int decimalPlaces)
	{
		double rndLocation	= Utilities.round(location,	decimalPlaces);
		double rndScale		= Utilities.round(scale,	decimalPlaces);
		double rndMin		= Utilities.round(min,		decimalPlaces);
		double rndMax		= Utilities.round(max,		decimalPlaces);
		return SHORT_ID + "(" + rndLocation + ", " + rndScale 
								+ ", " + rndMin + " < x < " + rndMax + ")";
	}

	@Override
	public double getMean()
	{
		return computeMean(location, scale, min, max);
	}

	@Override
	public double getStDev()
	{
		return computeStDev(location, scale, min, max);
	}

	@Override
	public double getVar()
	{
		return computeVar(location, scale, min, max);
	}
	
	@Override
	public double getSkewness()
	{
		// TODO Implement
		return Double.NaN;
	}
	
	@Override
	public double getpdf(double x)
	{
		return computepdf(location, scale, min, max, x);
	}

	@Override
	public double getCDF(double x)
	{
		return computeCDF(location, scale, min, max, x);
	}

	@Override
	public double getInvCDF(double p)
	{
		return computeInvCDF(location, scale, min, max, p);
	}

	@Override
	public TruncatedNormal truncate(double min, double max)
	{
		double lower 	= Math.max(min, this.min);
		double upper 	= Math.min(max, this.max);
		lower			= lower <= upper ? lower : upper;
		upper			= upper >= lower ? upper : lower;
		return new TruncatedNormal(location, scale, lower, upper);
	}

	@Override
	public double sample()
	{
		return sample(location, scale, min, max);
	}
	
	@Override
	public void shift(double newMean)
	{
		location += newMean - getMean();
	}

	@Override
	public void scale(double newStDev)
	{
		// TODO Implement
	}
	
	// --------------------------------------------------------------------------------------------
	// Static methods
	// --------------------------------------------------------------------------------------------
	
	/**
	 * @param location	{@link #location} 
	 * @param scale		{@link #scale}
	 * @param min		{@link #min}
	 * @param max		{@link #max}
	 * @return The mean value of a truncated normal distribution
	 */
	public static double computeMean(double location, double scale, double min, double max)
	{
		double alpha		= (min - location)/scale;
		double beta			= (max - location)/scale;
		double normalizer	= cdfStdNormal(beta) - cdfStdNormal(alpha);
		return location + scale*((stdNormal(alpha) - stdNormal(beta))/normalizer);
	}
	
	/**
	 * @param location	{@link #location} 
	 * @param scale		{@link #scale}
	 * @param min		{@link #min}
	 * @param max		{@link #max}
	 * @return The standard deviation of a truncated normal distribution
	 */
	public static double computeStDev(double location, double scale, double min, double max)
	{
		return Math.sqrt(computeVar(location, scale, min, max));
	}
	
	/**
	 * @param location	{@link #location} 
	 * @param scale		{@link #scale}
	 * @param min		{@link #min}
	 * @param max		{@link #max}
	 * @return The variance of a truncated normal distribution
	 */
	public static double computeVar(double location, double scale, double min, double max)
	{
		double alpha		= (min - location)/scale;
		double beta			= (max - location)/scale;
		double normalizer	= cdfStdNormal(beta) - cdfStdNormal(alpha);
		double temp1		= (alpha*stdNormal(alpha) - beta*stdNormal(beta))/normalizer;
		double temp2		= (stdNormal(alpha) - stdNormal(beta))/normalizer;
		return scale*scale*(1 + temp1 - temp2*temp2);
	}
	
	/**
	 * @param location	{@link #location} 
	 * @param scale		{@link #scale}
	 * @param min		{@link #min}
	 * @param max		{@link #max}
	 * @param z			The point at which to evaluate
	 * @return			The value of the probability density function of a truncated normal
	 * 					distribution
	 */
	public static double computepdf(double location, double scale, double min, double max, 
										double x)
	{
		if (x <= min || x >= max)
			return 0.0;
		
		double alpha		= (min - location)/scale;
		double beta			= (max - location)/scale;
		double normalizer	= cdfStdNormal(beta) - cdfStdNormal(alpha);
		double temp			= (x - location)/scale;
		return stdNormal(temp)/(scale*normalizer);
	}
	
	/**
	 * @param location	{@link #location} 
	 * @param scale		{@link #scale}
	 * @param min		{@link #min}
	 * @param max		{@link #max}
	 * @param z			The point at which to evaluate
	 * @return			The value of the cumulative density function of a truncated normal
	 * 					distribution
	 */
	public static double computeCDF(double location, double scale, double min, double max, 
										double x)
	{
		if (x <= min)
			return 0.0;
		if (x >= max)
			return 1.0;
		
		double alpha		= (min - location)/scale;
		double beta			= (max - location)/scale;
		double normalizer	= cdfStdNormal(beta) - cdfStdNormal(alpha);
		double temp			= (x - location)/scale;
		return (cdfStdNormal(temp) - cdfStdNormal(alpha))/normalizer;
	}
	
	/**
	 * @param location	{@link #location} 
	 * @param scale		{@link #scale}
	 * @param min		{@link #min}
	 * @param max		{@link #max}
	 * @param p			The value of the cumulative distribution function
	 * @return			The value of a random variable with a truncated normal distribution that 
	 * has the cumulative distribution function value that enters as a parameter
	 */
	public static double computeInvCDF(double location, double scale, double min, double max, 
										double p)
	{
		if (p < 0 || p > 1)
			return Double.NaN;
		
		double nonTrCDFMin	= Normal.computeCDF(location, scale, min);
		double nonTrCDFMax	= Normal.computeCDF(location, scale, max);
		double modP			= nonTrCDFMin + p*(nonTrCDFMax - nonTrCDFMin);
		return Normal.computeInvCDF(location, scale, modP);
	}
	
	/**
	 * @param location	{@link #location} 
	 * @param scale		{@link #scale}
	 * @param min		{@link #min}
	 * @param max		{@link #max}
	 * @return			A random number from a given truncated normal distribution 
	 */
	public static double sample(double location, double scale, double min, double max)
	{
		double nonTrCDFMin	= Normal.computeCDF(location, scale, min);
		double nonTrCDFMax	= Normal.computeCDF(location, scale, max);
		double seed			= Uniform.sample(nonTrCDFMin, nonTrCDFMax);
		return Normal.computeInvCDF(location, scale, seed);
	}

	/**
	 * @param x Point at which to evaluate the density function
	 * @return The probability density function of a standard normal distribution
	 */
	private static double stdNormal(double x)
	{
		double exponent	= -0.5*x*x;
		return DOUBLE_PI_SQRT_INV*Math.exp(exponent);
	}
	
	/**
	 * @param x Point at which to evaluate the cumulative density function
	 * @return The cumulative density function of a standard normal distribution
	 */
	private static double cdfStdNormal(double x)
	{
		return Normal.computeCDF(x);
	}
	
}
