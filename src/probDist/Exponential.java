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
 * This class represents Exponential probability distributions and defines static methods for 
 * making computations with Exponential probability distributions. Available methods include: <ul>
 * <li>Compute the probability of a range in a Exponential distribution
 * <li>Compute the CDF and inverse CDF functions for a Exponential distribution
 * <li>Generate random numbers with a Exponential distribution </ul>
 * @author Felipe Hernández
 */
public class Exponential extends ContProbDist
{
	
	// --------------------------------------------------------------------------------------------
	// Constants
	// --------------------------------------------------------------------------------------------
	
	/**
	 * Exponential distribution type String identifier
	 */
	public final static String ID = "Exponential";
	
	/**
	 * Exponential distribution type short String identifier
	 */
	public final static String SHORT_ID = "Exp";
	
	// --------------------------------------------------------------------------------------------
	// Attributes
	// --------------------------------------------------------------------------------------------
	
	/**
	 * The rate parameter of the Exponential distribution. If <i>lambda</i> is positive, the
	 * Exponential probability density function is defined in the positive direction. If 
	 * <i>lambda</i> is negative, the probability density function is reflected around the density 
	 * axis.
	 */
	private double lambda;
	
	/**
	 * The origin of the Exponential distribution. Usually it is set as zero. However, it can be 
	 * modified such that the random variable is distributed <i>X - location ~ Exp(lambda)</i>.
	 */
	private double location;
	
	// --------------------------------------------------------------------------------------------
	// Constructors
	// --------------------------------------------------------------------------------------------
	
	/**
	 * Creates an Exponential probability distribution
	 * @param lambda The rate parameter of the Exponential distribution. If lambda is positive, the
	 * Exponential probability density function is defined in the positive direction. If lambda is 
	 * negative, the probability density function is reflected around the density axis.
	 */
	public Exponential(double lambda)
	{
		type = ContProbDist.EXPONENTIAL;
		this.lambda = lambda;
		this.location = 0;
	}
	
	/**
	 * Creates an Exponential probability distribution
	 * @param lambda The rate parameter of the Exponential distribution. If lambda is positive, the
	 * Exponential probability density function is defined in the positive direction. If lambda is 
	 * negative, the probability density function is reflected around the density axis.
	 * @param location The origin of the Exponential distribution. Usually it is set as zero. 
	 * However, it can be modified such that the random variable is distributed 
	 * <i>X - location ~ Exp(lambda)</i>.
	 */
	public Exponential(double lambda, double location)
	{
		type = ContProbDist.EXPONENTIAL;
		this.lambda = lambda;
		this.location = location;
	}
	
	/**
	 * Creates an Exponential probability distribution from a set of observed data. The 
	 * distribution's parameters are computed by using the mean and standard deviation of the data
	 * @param sample The data points to build the distribution from
	 */
	public Exponential(ContSeries sample)
	{
		type				= ContProbDist.EXPONENTIAL;
		double mean			= sample.getMean();
		double stDev		= sample.getStDev();
		double skewness		= sample.getSkewness();
		double estimator	= 1/stDev;
		
		lambda				= estimator*(skewness >= 0.0 ? 1 : -1);
		location			= mean - 1/lambda;
	}
	
	/**
	 * Creates an Exponential probability distribution from a set of observed data. The lambda 
	 * parameter is computed by using the mean of the data
	 * @param sample The data points to build the distribution from
	 * @param location The origin of the Exponential distribution. Usually it is set as 0. However, 
	 * it can be modified such that the random variable is distributed 
	 * <i>X - location ~ Exp(lambda)</i>.
	 */
	public Exponential(ContSeries sample, double location)
	{
		type = ContProbDist.EXPONENTIAL;
		double mean = sample.getMean();
		lambda = 1/(mean - location);
		this.location = location;
	}
	
	// --------------------------------------------------------------------------------------------
	// Non-static methods
	// --------------------------------------------------------------------------------------------
	
	/**
	 * @return The rate parameter of the Exponential distribution. If lambda is positive, the
	 * Exponential probability density function is defined in the positive direction. If lambda is 
	 * negative, the probability density function is reflected around the density axis.
	 */
	public double getLambda()
	{
		return lambda;
	}
	
	/**
	 * Sets the rate parameter of the Exponential distribution. If lambda is positive, the
	 * Exponential probability density function is defined in the positive direction. If lambda is 
	 * negative, the probability density function is reflected around the density axis.
	 * @param lambda The value to set
	 */
	public void setLambda(double lambda)
	{
		this.lambda = lambda;
	}
	
	/**
	 * @return The origin of the Exponential distribution. Usually it is set as zero. However, 
	 * it can be modified such that the random variable is distributed 
	 * <i>X - location ~ Exp(lambda)</i>.
	 */
	public double getLocation()
	{
		return location;
	}
	
	/**
	 * Sets the origin of the Exponential distribution. Usually it is set as zero. However, 
	 * it can be modified such that the random variable is distributed 
	 * <i>X - location ~ Exp(lambda)</i>.
	 * @param location The value to set
	 */
	public void setLocation(double location)
	{
		this.location = location;
	}
	
	@Override
	public double getMean()
	{
		return location + 1/lambda;
	}
	
	@Override
	public double getStDev()
	{
		return Math.abs(1/lambda);
	}
	
	@Override
	public double getVar() 
	{
		return 1/(lambda*lambda);
	}
	
	@Override
	public double getSkewness()
	{
		return lambda >= 0.0 ? 2.0 : -2.0;
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
			return SHORT_ID + "(" + lambda + ")";
		else
			return SHORT_ID + "(" + lambda + ", " + location + ")";
	}

	@Override
	public String toString(int decimalPlaces) 
	{
		double roundedLambda	= Utilities.round(lambda, decimalPlaces);
		double roundedLocation	= Utilities.round(location, decimalPlaces);
		if(location == 0.0)
			return SHORT_ID + "(" + roundedLambda + ")";
		else
			return SHORT_ID + "(" + roundedLambda + ", " + roundedLocation + ")";
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
		return computepdf(lambda, location, x);
	}

	@Override
	public double getCDF(double x) 
	{
		return computeCDF(lambda, location, x);
	}

	@Override
	public double getInvCDF(double p) 
	{
		return computeInvCDF(lambda, location, p);
	}

	@Override
	public double sample() 
	{
		return sample(lambda, location);
	}
	
	@Override
	public void shift(double newMean)
	{
		location += newMean - getMean();
	}

	@Override
	public void scale(double newStDev)
	{
		lambda = (lambda > 0.0 ? 1 : -1)*(1/newStDev);
	}
	
	// --------------------------------------------------------------------------------------------
	// Static methods
	// --------------------------------------------------------------------------------------------

	/**
	 * Computes the mean value of a random variable with a given Exponential distribution
	 * @param lambda The rate parameter of the Exponential distribution. If lambda is positive, the
	 * Exponential probability density function is defined in the positive direction. If lambda is 
	 * negative, the probability density function is reflected around the density axis.
	 * @param location The origin of the Exponential distribution. Usually it is set as 0. However, 
	 * it can be modified such that the random variable is distributed 
	 * <i>X - location ~ Exp(lambda)</i>.
	 * @return The mean value of a random variable with a given Exponential distribution
	 */
	public static double computeMean(double lambda, double location)
	{
		return location + 1/lambda;
	}
	
	/**
	 * Computes the standard deviation of a random variable with a given Exponential distribution
	 * @param lambda The rate parameter of the Exponential distribution. If lambda is positive, the
	 * Exponential probability density function is defined in the positive direction. If lambda is 
	 * negative, the probability density function is reflected around the density axis.
	 * @return The standard deviation of a random variable with a given Exponential distribution
	 */
	public static double computeStDev(double lambda)
	{
		return Math.abs(1/lambda);
	}
	
	/**
	 * Computes the variance of a random variable with a given Exponential distribution
	 * @param lambda The rate parameter of the Exponential distribution. If lambda is positive, the
	 * Exponential probability density function is defined in the positive direction. If lambda is 
	 * negative, the probability density function is reflected around the density axis.
	 * @return The variance of a random variable with a given Exponential distribution
	 */
	public static double computeVar(double lambda)
	{
		double stDev = computeStDev(lambda);
		return stDev*stDev;
	}
	
	/**
	 * Computes the value of the probability density function of an Exponential probability
	 * distribution
	 * @param lambda The rate parameter of the Exponential distribution. If lambda is positive, the
	 * Exponential probability density function is defined in the positive direction. If lambda is 
	 * negative, the probability density function is reflected around the density axis.
	 * @param x The value at which to compute the probability density
	 * @return The value of the probability density function of an Exponential probability
	 * distribution
	 */
	public static double computepdf(double lambda, double x)
	{
		return computepdf(lambda, 0.0, x);
	}
	
	/**
	 * Computes the value of the probability density function of an Exponential probability
	 * distribution
	 * @param lambda The rate parameter of the Exponential distribution. If lambda is positive, the
	 * Exponential probability density function is defined in the positive direction. If lambda is 
	 * negative, the probability density function is reflected around the density axis.
	 * @param location The origin of the Exponential distribution. Usually it is set as zero. 
	 * However, it can be modified such that the random variable is distributed 
	 * <i>X - location ~ Exp(lambda)</i>.
	 * @param x The value at which to compute the probability density
	 * @return The value of the probability density function of an Exponential probability
	 * distribution
	 */
	public static double computepdf(double lambda, double location, double x)
	{
		if (lambda == 0.0 || Double.isInfinite(x))
			return 0.0;
		
		double tempX 		= x - location;
		double tempLambda	= lambda;;
		if (lambda < 0.0)
		{
			tempX 		= -tempX;
			tempLambda 	= -lambda;
		}
		if(tempX < 0.0)
			return 0.0;
		return tempLambda*Math.exp(-tempLambda*tempX);
	}
	
	/**
	 * Computes the value of the cumulative distribution function of an Exponential probability 
	 * distribution
	 * @param lambda The rate parameter of the Exponential distribution. If lambda is positive, the
	 * Exponential probability density function is defined in the positive direction. If lambda is 
	 * negative, the probability density function is reflected around the density axis.
	 * @param x The independent value
	 * @return The value or quantile
	 */
	public static double computeCDF(double lambda, double x)
	{
		return Weibull.computeCDF(1/lambda, 1, x);
	}
	
	/**
	 * Computes the value of the cumulative distribution function of an Exponential probability 
	 * distribution
	 * @param lambda The rate parameter of the Exponential distribution. If lambda is positive, the
	 * Exponential probability density function is defined in the positive direction. If lambda is 
	 * negative, the probability density function is reflected around the density axis.
	 * @param location The origin of the Exponential distribution. Usually it is set as zero. 
	 * However, it can be modified such that the random variable is distributed 
	 * <i>X - location ~ Exp(lambda)</i>.
	 * @param x The independent value
	 * @return The value or quantile
	 */
	public static double computeCDF(double lambda, double location, double x)
	{
		return Weibull.computeCDF(1/lambda, 1, location, x);
	}
	
	/**
	 * Computes the value of a random variable with an Exponential distribution that has the 
	 * cumulative distribution function value that enters as a parameter. That is, the inverse 
	 * cumulative distribution function value. Returns Double.NaN if value of the p parameter is 
	 * smaller than 0 or larger than 1.
	 * @param lambda The rate parameter of the Exponential distribution. If lambda is positive, the
	 * Exponential probability density function is defined in the positive direction. If lambda is 
	 * negative, the probability density function is reflected around the density axis.
	 * @param p The value of the cumulative distribution function
	 * @return The value of a random variable with an Exponential distribution that has the 
	 * cumulative distribution function value that enters as a parameter
	 */
	public static double computeInvCDF(double lambda, double p)
	{
		return Weibull.computeInvCDF(1/lambda, 1, p);
	}
	
	/**
	 * Computes the value of a random variable with an Exponential distribution that has the 
	 * cumulative distribution function value that enters as a parameter. That is, the inverse 
	 * cumulative distribution function value. Returns {@link java.lang.Double#NaN} if value of the 
	 * p parameter is smaller than 0 or larger than 1.
	 * @param lambda The rate parameter of the Exponential distribution. If lambda is positive, the
	 * Exponential probability density function is defined in the positive direction. If lambda is 
	 * negative, the probability density function is reflected around the density axis.
	 * @param location The origin of the Exponential distribution. Usually it is set as zero. 
	 * However, it can be modified such that the random variable is distributed 
	 * <i>X - location ~ Exp(lambda)</i>.
	 * @param p The value of the cumulative distribution function
	 * @return The value of a random variable with an Exponential distribution that has the 
	 * cumulative distribution function value that enters as a parameter
	 */
	public static double computeInvCDF(double lambda, double location, double p)
	{
		return Weibull.computeInvCDF(1/lambda, 1, location, p);
	}
	
	/**
	 * Computes the root mean square error from comparing the accumulated weight function of a 
	 * series of data points to the cumulative probability density function of an exponential 
	 * distribution. This indicators helps measure how well the distribution fits the data.
	 * @param lambda The rate parameter of the Exponential distribution. If lambda is positive, the
	 * Exponential probability density function is defined in the positive direction. If lambda is 
	 * negative, the probability density function is reflected around the density axis.
	 * @param location The origin of the Exponential distribution. Usually it is set as zero. 
	 * However, it can be modified such that the random variable is distributed 
	 * <i>X - location ~ Exp(lambda)</i>.
	 * @param values The data points to be compared
	 * @return The root mean square error from comparing the accumulated weight function of a 
	 * series of data points to the cumulative probability density function of an exponential 
	 * distribution
	 */
	public static double computeRMSE(double lambda, double location, ContSeries values)
	{
		Exponential exponential = new Exponential(lambda, location);
		return exponential.computeRMSE(values);
	}
	
	/**
	 * Generates a random number with a given Exponential distribution
	 * @param lambda The rate parameter of the Exponential distribution. If lambda is positive, the
	 * Exponential probability density function is defined in the positive direction. If lambda is 
	 * negative, the probability density function is reflected around the density axis.
	 * @return A random number with a given Exponential distribution
	 */
	public static double sample(double lambda)
	{
		return Weibull.sample(1/lambda, 1);
	}
	
	/**
	 * Generates a random number with a given Exponential distribution
	 * @param lambda The rate parameter of the Exponential distribution. If lambda is positive, the
	 * Exponential probability density function is defined in the positive direction. If lambda is 
	 * negative, the probability density function is reflected around the density axis.
	 * @param location The origin of the Exponential distribution. Usually it is set as zero. 
	 * However, it can be modified such that the random variable is distributed 
	 * <i>X - location ~ Exp(lambda)</i>.
	 * @return A random number with a given Exponential distribution
	 */
	public static double sample(double lambda, double location)
	{
		return Weibull.sample(1/lambda, 1, location);
	}
	
}
