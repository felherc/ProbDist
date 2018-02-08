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
 * This class represents chi-squared distributions and defines static methods for making 
 * computations with chi-squared probability distributions. Available methods include: <ul>
 * <li>Compute the probability of a range in a chi-squared distribution
 * <li>Compute the CDF and inverse CDF functions for a chi-squared distribution
 * <li>Generate chi-squared distributed random numbers </ul>
 * @author Felipe Hernández
 */
public class ChiSquared extends ContProbDist
{
	
	// --------------------------------------------------------------------------------------------
	// Constants
	// --------------------------------------------------------------------------------------------
	
	/**
	 * Chi-squared distribution type String identifier
	 */
	public final static String ID = "Chi-squared";
	
	/**
	 * Chi-squared distribution type short String identifier
	 */
	public final static String SHORT_ID = "Chi2";
	
	/**
	 * The string identifier of the k parameter
	 */
	public final static String PARAM_K = "k";
	
	/**
	 * The string identifier of the location parameter
	 */
	public final static String PARAM_LOCATION = "location";
	
	// --------------------------------------------------------------------------------------------
	// Attributes
	// --------------------------------------------------------------------------------------------
	
	/**
	 * The number of degrees of freedom of the chi-squared distribution. If <i<>k</i> is positive,
	 * the chi-squared probability density function is defined in the positive direction. If 
	 * <i<>k</i> is negative, the probability density function is reflected around the 
	 * <i>location</i> axis.
	 */
	private double k;
	
	/**
	 * The origin of the chi-squared distribution. Usually it is set as zero. However, it can be 
	 * modified such that the random variable is distributed <i>X - location ~ Chi2(k)</i>.
	 */
	private double location;
	
	// --------------------------------------------------------------------------------------------
	// Constructors
	// --------------------------------------------------------------------------------------------
	
	/**
	 * Creates a chi-squared probability distribution
	 * @param k The number of degrees of freedom of the chi-squared distribution. If <i<>k</i> is 
	 * positive, the chi-squared probability density function is defined in the positive direction. 
	 * If <i<>k</i> is negative, the probability density function is reflected around the 
	 * <i>location</i> axis.
	 */
	public ChiSquared(double k)
	{
		type = ContProbDist.CHI_SQUARE;
		this.k = k;
		location = 0;
	}
	
	/**
	 * Creates a chi-squared probability distribution
	 * @param k The number of degrees of freedom of the chi-squared distribution. If <i<>k</i> is 
	 * positive, the chi-squared probability density function is defined in the positive direction. 
	 * If <i<>k</i> is negative, the probability density function is reflected around the 
	 * <i>location</i> axis.
	 * @param location The origin of the chi-squared distribution. Usually it is set as zero. 
	 * However, it can be modified such that the random variable is distributed 
	 * <i>X - location ~ Chi2(k)</i>.
	 */
	public ChiSquared(double k, double location)
	{
		type = ContProbDist.CHI_SQUARE;
		this.k = k;
		this.location = location;
	}
	
	/**
	 * Creates a chi-squared probability distribution from a set of observed data. The 
	 * distribution's parameters are computed by using the mean and standard deviation of the data
	 * @param sample The data points to build the distribution from
	 */
	public ChiSquared(ContSeries sample)
	{
		type				= ContProbDist.CHI_SQUARE;
		double mean			= sample.getMean();
		double stDev		= sample.getStDev();
		double skewness		= sample.getSkewness();
		double estimator	= stDev*stDev/2;
		
		k					= estimator*(skewness >= 0.0 ? 1 : -1);
		location			= mean - k;
	}
	
	/**
	 * Creates a chi-squared probability distribution from a set of observed data. The <i>k</i> 
	 * parameter is computed by using the mean of the data
	 * @param sample The data points to build the distribution from
	 * @param location The origin of the chi-squared distribution. Usually it is set as zero. 
	 * However, it can be modified such that the random variable is distributed 
	 * <i>X - location ~ Chi2(k)</i>.
	 */
	public ChiSquared(ContSeries sample, double location)
	{
		type = ContProbDist.CHI_SQUARE;
		double mean = sample.getMean();
		k = mean - location;
		this.location = location;
	}
	
	// --------------------------------------------------------------------------------------------
	// Non-static methods
	// --------------------------------------------------------------------------------------------

	/**
	 * @return The number of degrees of freedom of the chi-squared distribution. If <i<>k</i> is 
	 * positive, the chi-squared probability density function is defined in the positive direction. 
	 * If <i<>k</i> is negative, the probability density function is reflected around the 
	 * <i>location</i> axis.
	 */
	public double getK()
	{
		return k;
	}
	
	/**
	 * @param k The number of degrees of freedom of the chi-squared distribution. If <i<>k</i> is 
	 * positive, the chi-squared probability density function is defined in the positive direction. 
	 * If <i<>k</i> is negative, the probability density function is reflected around the 
	 * <i>location</i> axis.
	 */
	public void setK(double k)
	{
		this.k = k;
	}
	
	/**
	 * @return The origin of the chi-squared distribution. Usually it is set as zero. However, it 
	 * can be modified such that the random variable is distributed <i>X - location ~ Chi2(k)</i>.
	 */
	public double getLocation() 
	{
		return location;
	}

	/**
	 * @param location The origin of the chi-squared distribution. Usually it is set as zero. 
	 * However, it can be modified such that the random variable is distributed 
	 * <i>X - location ~ Chi2(k)</i>.
	 */
	public void setLocation(double location) 
	{
		this.location = location;
	}

	@Override
	public double getMean() 
	{
		return location + k;
	}

	@Override
	public double getStDev() 
	{
		return Math.sqrt(2*Math.abs(k));
	}
	
	@Override
	public double getVar() 
	{
		return 2*Math.abs(k);
	}
	
	@Override
	public double getSkewness()
	{
		double base = Math.sqrt(8/Math.abs(k));
		return k >= 0.0 ? base : -base;
	}
	
	@Override
	public String getTypeString() 
	{
		return ID;
	}

	@Override
	public String toString() 
	{
		if(location == 0.0)
			return SHORT_ID + "(" + k + ")";
		else
			return SHORT_ID + "(" + k + ", " + location + ")";
	}

	@Override
	public String toString(int decimalPlaces) 
	{
		double roundedK = Utilities.round(k, decimalPlaces);
		double roundedLocation = Utilities.round(location, decimalPlaces);
		if(location == 0.0)
			return SHORT_ID + "(" + roundedK + ")";
		else
			return SHORT_ID + "(" + roundedK + ", " + roundedLocation + ")";
	}
	
	@Override
	public ContProbDist truncate(double min, double max)
	{
		// TODO Implement
		return null;
	}
	
	@Override
	public double getpdf(double x) 
	{
		return computepdf(k, location, x);
	}

	@Override
	public double getCDF(double x) 
	{
		return computeCDF(k, location, x);
	}

	@Override
	public double getInvCDF(double p) 
	{
		return computeInvCDF(k, location, p);
	}

	@Override
	public double sample() 
	{
		return sample(k, location);
	}
	
	@Override
	public void shift(double newMean)
	{
		location += newMean - getMean();
	}

	@Override
	public void scale(double newStDev)
	{
		k = (k > 0.0 ? 1 : -1)*0.5*newStDev*newStDev;
	}
	
	// --------------------------------------------------------------------------------------------
	// Static methods
	// --------------------------------------------------------------------------------------------

	/**
	 * Computes the mean value of a random variable with a given chi-squared probability 
	 * distribution
	 * @param k The number of degrees of freedom of the chi-squared distribution. If <i<>k</i> is 
	 * positive, the chi-squared probability density function is defined in the positive direction. 
	 * If <i<>k</i> is negative, the probability density function is reflected around the 
	 * <i>location</i> axis.
	 * @param location The origin of the chi-squared distribution. Usually it is set as zero. 
	 * However, it can be modified such that the random variable is distributed 
	 * <i>X - location ~ Chi2(k)</i>.
	 * @return The mean value of a random variable with a given chi-squared probability 
	 * distribution
	 */
	public static double computeMean(double k, double location)
	{
		return location + k;
	}
	
	/**
	 * Computes the standard deviation of a random variable with a given chi-squared probability 
	 * distribution
	 * @param k The number of degrees of freedom of the chi-squared distribution. If <i<>k</i> is 
	 * positive, the chi-squared probability density function is defined in the positive direction. 
	 * If <i<>k</i> is negative, the probability density function is reflected around the 
	 * <i>location</i> axis.
	 * @return The standard deviation of a random variable with a given chi-squared probability 
	 * distribution
	 */
	public static double computeStDev(double k)
	{
		return Math.sqrt(2*k);
	}
	
	/**
	 * Computes the value of the probability density function of a chi-squared probability 
	 * distribution
	 * @param k The number of degrees of freedom of the chi-squared distribution. If <i<>k</i> is 
	 * positive, the chi-squared probability density function is defined in the positive direction. 
	 * If <i<>k</i> is negative, the probability density function is reflected around the 
	 * <i>location</i> axis.
	 * <i>X - location ~ Chi2(k)</i>.
	 * @param x The value at which to compute the probability density
	 * @return The value of the probability density function of a chi-squared probability 
	 * distribution
	 */
	public static double computepdf(double k, double x)
	{
		return computepdf(k, 0.0, x);
	}
	
	/**
	 * Computes the value of the probability density function of a chi-squared probability 
	 * distribution
	 * @param k The number of degrees of freedom of the chi-squared distribution. If <i<>k</i> is 
	 * positive, the chi-squared probability density function is defined in the positive direction. 
	 * If <i<>k</i> is negative, the probability density function is reflected around the 
	 * <i>location</i> axis.
	 * @param location The origin of the chi-squared distribution. Usually it is set as zero. 
	 * However, it can be modified such that the random variable is distributed 
	 * <i>X - location ~ Chi2(k)</i>.
	 * @param x The value at which to compute the probability density
	 * @return The value of the probability density function of a chi-squared probability 
	 * distribution
	 */
	public static double computepdf(double k, double location, double x)
	{
		x = x - location;
		
		if (k == 0.0 || Double.isInfinite(x))
			if(x == 0.0)
				return Double.POSITIVE_INFINITY;
			else
				return 0.0;
		
		double tempX 	= k > 0 ? x : -x;
		double tempK 	= Math.abs(k);
		if (tempX < 0.0)
			return 0.0;
		double num		= Math.pow(tempX, 0.5*tempK - 1)*Math.exp(-0.5*tempX);
		double den		= Math.pow(2, 0.5*tempK)*Gamma.computeGamma(0.5*tempK);
		return num/den;
	}
	
	/**
	 * Computes the value of the cumulative density function of a chi-squared probability 
	 * distribution
	 * @param k The number of degrees of freedom of the chi-squared distribution. If <i<>k</i> is 
	 * positive, the chi-squared probability density function is defined in the positive direction. 
	 * If <i<>k</i> is negative, the probability density function is reflected around the 
	 * <i>location</i> axis.
	 * @param x The independent value
	 * @return The value or quantile
	 */
	public static double computeCDF(double k, double x)
	{
		return computeCDF(k, 0.0, x);
	}
	
	/**
	 * Computes the value of the cumulative density function of a chi-squared probability 
	 * distribution
	 * @param k The number of degrees of freedom of the chi-squared distribution. If <i<>k</i> is 
	 * positive, the chi-squared probability density function is defined in the positive direction. 
	 * If <i<>k</i> is negative, the probability density function is reflected around the 
	 * <i>location</i> axis.
	 * @param location The origin of the chi-squared distribution. Usually it is set as zero. 
	 * However, it can be modified such that the random variable is distributed 
	 * <i>X - location ~ Chi2(k)</i>.
	 * @param x The independent value
	 * @return The value or quantile
	 */
	public static double computeCDF(double k, double location, double x)
	{
		x = x - location;
		if(k == 0 || Double.isInfinite(x))
			if(x < 0)
				return 0;
			else
				return 1;
		else if(k > 0)
		{
			if(x == 0)
				return 0;
			else
				return x < 0 ? 0 : Gamma.computeCDF(k/2.0, 2.0, x);
		}
		else
		{
			if(x == 0)
				return 1;
			else
				return x > 0 ? 1 : 1.0 - Gamma.computeCDF(-k/2.0, 2.0, -x);
		}
	}
	
	/**
	 * Computes the value of a random variable with a chi-squared distribution that has the 
	 * cumulative distribution function value that enters as a parameter. That is, the inverse 
	 * cumulative distribution function value. The <i>p</i> parameter must be within the closed
	 * interval <i>[0, 1]</i>.
	 * @param k The number of degrees of freedom of the chi-squared distribution. If <i<>k</i> is 
	 * positive, the chi-squared probability density function is defined in the positive direction. 
	 * If <i<>k</i> is negative, the probability density function is reflected around the 
	 * <i>location</i> axis.
	 * @param p The value of the cumulative distribution function
	 * @return The value of a random variable with a chi-squared distribution that has the 
	 * cumulative distribution function value that enters as a parameter
	 */
	public static double computeInvCDF(double k, double p)
	{
		return computeInvCDF(k, 0, p);
	}
	
	/**
	 * Computes the value of a random variable with a chi-squared distribution that has the 
	 * cumulative distribution function value that enters as a parameter. That is, the inverse 
	 * cumulative distribution function value. The <i>p</i> parameter must be within the closed
	 * interval <i>[0, 1]</i>.
	 * @param k The number of degrees of freedom of the chi-squared distribution. If <i<>k</i> is 
	 * positive, the chi-squared probability density function is defined in the positive direction. 
	 * If <i<>k</i> is negative, the probability density function is reflected around the 
	 * <i>location</i> axis.
	 * @param location The origin of the chi-squared distribution. Usually it is set as zero. 
	 * However, it can be modified such that the random variable is distributed 
	 * <i>X - location ~ Chi2(k)</i>.
	 * @param p The value of the cumulative distribution function
	 * @return The inverse cumulative distribution function value
	 */
	public static double computeInvCDF(double k, double location, double p)
	{
		if(p < 0 || p > 1)
			throw new IllegalArgumentException("the probability p must be within the range " +
												"[0, 1]");
		if(k == 0.0)
			return location;
		
		boolean positiveK = k > 0;
		if(p == 1.0)
			return positiveK ? Double.POSITIVE_INFINITY : location;
		if(p == 0.0)
			return positiveK ? location : Double.NEGATIVE_INFINITY;
		
		k = Math.abs(k);
		p = positiveK ? p : 1.0 - p;
		
		double centPos = Gamma.computeInvCDF(k/2.0, 2.0, p);
		return (positiveK ? centPos : -centPos) + location;
	}
	
	/**
	 * Generates a random number with a given chi-squared distribution
	 * @param k The number of degrees of freedom of the chi-squared distribution. If <i<>k</i> is 
	 * positive, the chi-squared probability density function is defined in the positive direction. 
	 * If <i<>k</i> is negative, the probability density function is reflected around the 
	 * <i>location</i> axis.
	 * <i>X - location ~ Chi2(k)</i>.
	 * @return A random number with a given chi-squared distribution
	 */
	public static double sample(double k)
	{
		return sample(k, 0);
	}
	
	/**
	 * Generates a random number with a given chi-squared distribution
	 * @param k The number of degrees of freedom of the chi-squared distribution. If <i<>k</i> is 
	 * positive, the chi-squared probability density function is defined in the positive direction. 
	 * If <i<>k</i> is negative, the probability density function is reflected around the 
	 * <i>location</i> axis.
	 * @param location The origin of the chi-squared distribution. Usually it is set as zero. 
	 * However, it can be modified such that the random variable is distributed 
	 * <i>X - location ~ Chi2(k)</i>.
	 * @return A random number with a given chi-squared distribution
	 */
	public static double sample(double k, double location)
	{
		double absK = Math.abs(k);
		int toGenerate = (int)Math.ceil(absK);
		double sum = 0;
		for(int i = 0 ; i < toGenerate ; i++)
		{
			double multiplier = i < (absK - 1) ? 1 : absK%1;
			double randomNormal = multiplier*Normal.sample(0, 1);
			sum+= randomNormal*randomNormal;
		}
		return (k/absK)*sum + location;
	}

}
