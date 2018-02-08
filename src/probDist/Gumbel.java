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
 * This class represents Gumbel probability distributions (Generalized Extreme Value Type-I 
 * distributions)
 * @author Felipe Hernández
 */
public class Gumbel extends ContProbDist
{

	// --------------------------------------------------------------------------------------------
	// Constants
	// --------------------------------------------------------------------------------------------
	
	/**
	 * Gumbel distribution type String identifier
	 */
	public final static String ID = "Gumbel";
	
	/**
	 * Gumbel distribution type short String identifier
	 */
	public final static String SHORT_ID = "Gum";
	
	/**
	 * Euler-Mascheroni constant used in several computations of the distribution
	 */
	public final static double EULER_MASCHERONI_CONSTANT = 
											0.5772156649015328606065120900824024310421593359399;
	
	// --------------------------------------------------------------------------------------------
	// Attributes
	// --------------------------------------------------------------------------------------------
	
	/**
	 * The mode of the Gumbel distribution, that is, the value where the distribution has the 
	 * highest probability density. Usually represented by the Greek letter miu.
	 */
	private double mode;
	
	/**
	 * The spread of the distribution. If positive, the Gumbel probability density function is 
	 * defined in the positive direction. If negative, the probability density function is 
	 * reflected around the density axis. Usually represented by the Greek letter beta.
	 */
	private double scale;
	
	// --------------------------------------------------------------------------------------------
	// Constructors
	// --------------------------------------------------------------------------------------------
	
	/**
	 * @param scale {@link #scale}
	 */
	public Gumbel(double scale)
	{
		type		= ContProbDist.GUMBEL;
		this.scale	= scale;
	}
	
	/**
	 * @param mode	{@link #mode}
	 * @param scale		{@link #scale}
	 */
	public Gumbel(double mode, double scale)
	{
		type			= ContProbDist.GUMBEL;
		this.mode		= mode;
		this.scale		= scale;
	}
	
	/**
	 * Creates a Gumbel probability distribution from a set of observed data. The distribution's 
	 * parameters are computed by using the mean and standard deviation of the data
	 * @param sample The data points to build the distribution from
	 */
	public Gumbel(ContSeries sample)
	{
		type			= ContProbDist.GUMBEL;
		double mean		= sample.getMean();
		double var		= sample.getVar();
		double skewness	= sample.getSkewness();
		
		scale			= Math.sqrt(6*var)/Math.PI*(skewness >= 0.0 ? 1 : -1);
		mode			= mean - EULER_MASCHERONI_CONSTANT*scale;
	}
	
	// --------------------------------------------------------------------------------------------
	// Non-static methods
	// --------------------------------------------------------------------------------------------
	
	/**
	 * @return {@link #mode}
	 */
	public double getMode()
	{
		return mode;
	}

	/**
	 * @param mode {@link #mode}
	 */
	public void setMode(double mode)
	{
		this.mode	= mode;
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
		this.scale	= scale;
	}

	@Override
	public String getTypeString()
	{
		return ID;
	}

	@Override
	public String toString()
	{
		if(mode == 0.0)
			return SHORT_ID + "(" + scale + ")";
		else
			return SHORT_ID + "(" + scale + ", " + mode + ")";
	}

	@Override
	public String toString(int decimalPlaces)
	{
		double roundedScale	= Utilities.round(scale,	decimalPlaces);
		double roundedMode	= Utilities.round(mode,		decimalPlaces);
		if(mode == 0.0)
			return SHORT_ID + "(" + roundedScale + ")";
		else
			return SHORT_ID + "(" + roundedScale + ", " + roundedMode + ")";
	}

	@Override
	public double getMean()
	{
		return computeMean(mode, scale);
	}

	@Override
	public double getStDev()
	{
		return computeStDev(mode, scale);
	}

	@Override
	public double getVar()
	{
		return computeVariance(mode, scale);
	}
	
	@Override
	public double getSkewness()
	{
		return scale >= 0.0 ? 1.14 : -1.14;
	}

	@Override
	public ContProbDist truncate(double min, double max)
	{
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public double getpdf(double x)
	{
		return computepdf(mode, scale, x);
	}

	@Override
	public double getCDF(double x)
	{
		return computeCDF(mode, scale, x);
	}

	@Override
	public double getInvCDF(double p)
	{
		return computeInvCDF(mode, scale, p);
	}

	@Override
	public double sample()
	{
		return sample(mode, scale);
	}
	
	@Override
	public void shift(double newMean)
	{
		mode += newMean - getMean();
	}

	@Override
	public void scale(double newStDev)
	{
		scale = (scale >= 1.0 ? 1 : -1)*Math.sqrt(6)*newStDev/Math.PI;
	}
	
	// --------------------------------------------------------------------------------------------
	// Static methods
	// --------------------------------------------------------------------------------------------
	
	/**
	 * @param mode	{@link #mode}
	 * @param scale	{@link #scale}
	 * @return The mean value of a random variable with a given Gumbel distribution
	 */
	public static double computeMean(double mode, double scale)
	{
		return mode + scale*EULER_MASCHERONI_CONSTANT;
	}
	
	/**
	 * @param mode	{@link #mode}
	 * @param scale	{@link #scale}
	 * @return The standard deviation of a random variable with a given Gumbel distribution
	 */
	public static double computeStDev(double mode, double scale)
	{
		return Math.PI*Math.abs(scale)/Math.sqrt(6);
	}
	
	/**
	 * @param mode	{@link #mode}
	 * @param scale	{@link #scale}
	 * @return The variance of a random variable with a given Gumbel distribution
	 */
	public static double computeVariance(double mode, double scale)
	{
		double pi = Math.PI;
		return pi*pi*scale*scale/6;
	}
	
	/**
	 * @param mode	{@link #mode}
	 * @param scale	{@link #scale}
	 * @param x		The value at which to compute the probability density
	 * @return The value of the probability density function of a Gumbel probability distribution
	 */
	public static double computepdf(double mode, double scale, double x)
	{
		if (scale >= 0.0)
		{
			double z		= (x - mode)/scale;
			return (Math.exp(-(z + Math.exp(-z))))/scale;
		}
		else
		{
			double z		= (x - mode)/scale;
			return -(Math.exp(-(z + Math.exp(-z))))/scale;
		}
	}
	
	/**
	 * Computes the value of the cumulative distribution function of a Gumbel probability 
	 * distribution
	 * @param mode	{@link #mode}
	 * @param scale	{@link #scale}
	 * @param x		The value at which to compute the cumulative probability density
	 * @return The value or quantile
	 */
	public static double computeCDF(double mode, double scale, double x)
	{
		double z = (x - mode)/scale;
		if (scale >= 0.0)
			return Math.exp(-Math.exp(-z));
		else
			return 1 - Math.exp(-Math.exp(-z));
	}
	
	/**
	 * Computes the value of a random variable with a Gumbel distribution that has the cumulative 
	 * distribution function value that enters as a parameter. That is, the inverse cumulative 
	 * distribution function value. Returns Double.NaN if value of the p parameter is smaller than 
	 * 0 or larger than 1.
	 * @param mode	{@link #mode}
	 * @param scale	{@link #scale}
	 * @param p		The value of the cumulative distribution function
	 * @return		The value of a random variable with an Exponential distribution that has the 
	 * 				cumulative distribution function value that enters as a parameter
	 */
	public static double computeInvCDF(double mode, double scale, double p)
	{
		if (p < 0.0 || p > 1.0)
			return Double.NaN;
		double p2	= scale >= 0.0 ? p : 1 - p;
		return mode - scale*Math.log(Math.log(1/p2));
	}
	
	/**
	 * Generates a random number with a given Gumbel distribution
	 * @param mode	{@link #mode}
	 * @param scale	{@link #scale}
	 * @return A random number with a given Gumbel distribution
	 */
	public static double sample(double mode, double scale)
	{
		double uniform = 1 - Math.random();
		return mode - scale*Math.log(Math.log(1/uniform));
	}

}
