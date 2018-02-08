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
 * This class represents Log-normal probability distributions and defines static methods for 
 * making computations with Log-normal probability distributions. Available methods include: <ul>
 * <li>Compute the probability of a range in a Log-normal distribution
 * <li>Compute the CDF and inverse CDF functions for a Log-normal distribution
 * <li>Generate random numbers with a Log-normal distribution </ul>
 * @author Felipe Hernández
 */
public class LogNormal extends ContProbDist
{
	
	// --------------------------------------------------------------------------------------------
	// Constants
	// --------------------------------------------------------------------------------------------
	
	/**
	 * Log-normal distribution type String identifier
	 */
	public final static String ID = "Log-normal";
	
	/**
	 * Log-normal distribution type short String identifier
	 */
	public final static String SHORT_ID = "Log-N";
	
	// --------------------------------------------------------------------------------------------
	// Attributes
	// --------------------------------------------------------------------------------------------
	
	/**
	 * The mean of the underlying normal distribution
	 */
	private double miu;
	
	/**
	 * The standard deviation of the underlying normal distribution
	 */
	private double sigma;
	
	/**
	 * The origin of the Log-normal distribution. Usually it is set as zero. However, it can be 
	 * modified such that the random variable is distributed <i>X - location ~ Log-N(...)</i>.
	 */
	private double location;
	
	/**
	 * <code>True</code> if the distribution is defined from the {@link # location} in the positive
	 * direction. <code>False</code> if it is defined in the negative direction.
	 */
	private boolean positiveDir;
	
	// --------------------------------------------------------------------------------------------
	// Constructors
	// --------------------------------------------------------------------------------------------
	
	/**
	 * Creates a Log-normal probability distribution. Assumes a standard location of 0 and a 
	 * positive direction.
	 * @param miu	{@link #miu}
	 * @param sigma	{@link #sigma}
	 */
	public LogNormal(double miu, double sigma)
	{
		type		= ContProbDist.LOG_NORMAL;
		this.miu	= miu;
		this.sigma	= sigma;
		location	= 0.0;
		positiveDir	= true;
	}
	
	/**
	 * Creates a Log-normal probability distribution
	 * @param miu			{@link #miu}
	 * @param sigma			{@link #sigma}
	 * @param location		{@link #location}
	 * @param positiveDir	{@link #positiveDir}
	 */
	public LogNormal(double miu, double sigma, double location, boolean positiveDir)
	{
		type				= ContProbDist.LOG_NORMAL;
		this.miu			= miu;
		this.sigma			= sigma;
		this.location		= location;
		this.positiveDir	= positiveDir;
	}
	
	/**
	 * Creates a Log-normal probability distribution
	 * @param sample	The data points to build the distribution from	
	 * @param location	{@link #location}
	 */
	public LogNormal(ContSeries sample, double location)
	{
		type				= ContProbDist.LOG_NORMAL;
		this.location		= location;
		double mean			= sample.getMean();
		positiveDir			= location < mean ? true : false;
		double var			= sample.getVar();
		double m			= Math.abs(mean - location);
		double temp			= 1 + var/(m*m);
		miu					= Math.log(m/(Math.sqrt(temp)));
		sigma				= Math.sqrt(Math.log(temp));
	}
	
	/**
	 * Creates a Log-normal probability distribution
	 * @param sample The data points to build the distribution from
	 */
	public LogNormal(ContSeries sample)
	{
		type				= ContProbDist.LOG_NORMAL;
		// TODO Implement
	}
	
	// --------------------------------------------------------------------------------------------
	// Non-static methods
	// --------------------------------------------------------------------------------------------

	/**
	 * @return {@link #miu}
	 */
	public double getMiu()
	{
		return miu;
	}
	
	/**
	 * @param miu {@link #miu}
	 */
	public void setMiu(double miu)
	{
		this.miu = miu;
	}
	
	/**
	 * @return {@link #sigma}
	 */
	public double getSigma()
	{
		return sigma;
	}
	
	/**
	 * @param sigma {@link #sigma}
	 */
	public void setSigma(double sigma)
	{
		this.sigma = sigma;
	}
	
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
	 * @return {@link #positiveDir}
	 */
	public boolean hasPositiveDir()
	{
		return positiveDir;
	}
	
	/**
	 * @param positiveDir {@link #positiveDir}
	 */
	public void setPositiveDir(boolean positiveDir)
	{
		this.positiveDir = positiveDir;
	}
	
	@Override
	public double getMean()
	{
		return computeMean(miu, sigma, location, positiveDir);
	}

	@Override
	public double getStDev()
	{
		return computeStDev(miu, sigma);
	}

	@Override
	public double getVar()
	{
		return computeVar(miu, sigma);
	}
	
	@Override
	public double getSkewness()
	{
		double sigma2	= sigma*sigma;
		double base		= (Math.exp(sigma2) + 2)*Math.sqrt(Math.exp(sigma2) - 1);
		return positiveDir ? base : -base;
	}

	@Override
	public String getTypeString()
	{
		return ID;
	}

	@Override
	public String toString()
	{
		String dir		= positiveDir ? "" : ", neg";
		if(location == 0.0)
			return SHORT_ID + "(" + miu + ", " + sigma +					dir + ")";
		else
			return SHORT_ID + "(" + miu + ", " + sigma + ", " + location +	dir + ")";
	}

	@Override
	public String toString(int decimalPlaces)
	{
		double rMiu			= Utilities.round(miu);
		double rSigma		= Utilities.round(sigma);
		double rLocation	= Utilities.round(location);
		String dir			= positiveDir ? "" : ", neg";
		if(location == 0.0)
			return SHORT_ID + "(" + rMiu + ", " + rSigma +						dir + ")";
		else
			return SHORT_ID + "(" + rMiu + ", " + rSigma + ", " + rLocation +	dir + ")";
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
		return computepdf(miu, sigma, location, positiveDir, x);
	}

	@Override
	public double getCDF(double x)
	{
		return computeCDF(miu, sigma, location, positiveDir, x);
	}

	@Override
	public double getInvCDF(double p)
	{
		return computeInvCDF(miu, sigma, location, positiveDir, p);
	}

	@Override
	public double sample()
	{
		return sample(miu, sigma, location, positiveDir);
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
	 * Computes the mean value of a random variable with a given Log-normal distribution. Assumes
	 * a standard location of 0 and a positive direction.
	 * @param miu	The mean of the underlying normal distribution
	 * @param sigma	The standard deviation of the underlying normal distribution
	 * @return The mean value of a random variable with a given Log-normal distribution
	 */
	public static double computeMean(double miu, double sigma)
	{
		return computeMean(miu, sigma, 0.0, true);
	}
	
	/**
	 * Computes the mean value of a random variable with a given Log-normal distribution
	 * @param miu			The mean of the underlying normal distribution
	 * @param sigma			The standard deviation of the underlying normal distribution
	 * @param location		The origin of the Log-normal distribution. Usually it is set as zero. 
	 * 						However, it can be modified such that the random variable is 
	 * 						distributed <i>X - location ~ Log-N(...)</i>.
	 * @param positiveDir	<code>True</code> if the distribution is defined from the 
	 * 						{@link # location} in the positive direction. <code>False</code> if it 
	 * 						is defined in the negative direction.
	 * @return The mean value of a random variable with a given Log-normal distribution
	 */
	public static double computeMean(double miu, double sigma, double location, 
										boolean positiveDir)
	{
		double sigma2	= sigma*sigma;
		double standard	= Math.exp(miu + sigma2/2);
		int direction	= positiveDir ? 1 : -1;
		return location + direction*standard;
	}
	
	/**
	 * Computes the standard deviation of a random variable with a given Log-normal distribution
	 * @param miu	The mean of the underlying normal distribution
	 * @param sigma	The standard deviation of the underlying normal distribution
	 * @return The standard deviation of a random variable with a given Log-normal distribution
	 */
	public static double computeStDev(double miu, double sigma)
	{
		return Math.sqrt(computeVar(miu, sigma));
	}
	
	/**
	 * Computes the variance of a random variable with a given Log-normal distribution
	 * @param miu	The mean of the underlying normal distribution
	 * @param sigma	The standard deviation of the underlying normal distribution
	 * @return The variance of a random variable with a given Log-normal distribution
	 */
	public static double computeVar(double miu, double sigma)
	{
		double sigma2	= sigma*sigma;
		double temp1	= Math.exp(sigma2) - 1;
		double temp2	= Math.exp(2*miu + sigma2);
		return temp1*temp2;
	}
	
	/**
	 * Computes the value of the probability density function of an Log-normal probability
	 * distribution. Assumes a standard location of 0 and a positive direction.
	 * @param miu	The mean of the underlying normal distribution
	 * @param sigma	The standard deviation of the underlying normal distribution
	 * @param x		The value at which to compute the probability density
	 * @return The value of the probability density function of an Log-normal probability
	 */
	public static double computepdf(double miu, double sigma, double x)
	{
		return computepdf(miu, sigma, 0.0, true, x);
	}
	
	/**
	 * Computes the value of the probability density function of an Log-normal probability
	 * distribution
	 * @param miu			The mean of the underlying normal distribution
	 * @param sigma			The standard deviation of the underlying normal distribution
	 * @param location		The origin of the Log-normal distribution. Usually it is set as zero. 
	 * 						However, it can be modified such that the random variable is 
	 * 						distributed <i>X - location ~ Log-N(...)</i>.
	 * @param positiveDir	<code>True</code> if the distribution is defined from the 
	 * 						{@link # location} in the positive direction. <code>False</code> if it 
	 * 						is defined in the negative direction.
	 * @param x				The value at which to compute the probability density
	 * @return The value of the probability density function of an Log-normal probability
	 */
	public static double computepdf(double miu, double sigma, double location, 
										boolean positiveDir, double x)
	{
		int direction	= positiveDir ? 1 : -1;
		if (direction*(x - location) < 0.0)
			return 0.0;
		double z		= direction*(x - location);
		double xtrans	= Math.log(z);
		return Normal.computepdf(miu, sigma, xtrans)/z;
	}
	
	/**
	 * Computes the value of the cumulative distribution function of an Log-normal probability 
	 * distribution. Assumes a standard location of 0 and a positive direction.
	 * @param miu	The mean of the underlying normal distribution
	 * @param sigma	The standard deviation of the underlying normal distribution
	 * @param x		The independent value
	 * @return The value or quantile
	 */
	public static double computeCDF(double miu, double sigma, double x)
	{
		return computeCDF(miu, sigma, 0.0, true, x);
	}
	
	/**
	 * Computes the value of the cumulative distribution function of an Log-normal probability 
	 * distribution
	 * @param miu			The mean of the underlying normal distribution
	 * @param sigma			The standard deviation of the underlying normal distribution
	 * @param location		The origin of the Log-normal distribution. Usually it is set as zero. 
	 * 						However, it can be modified such that the random variable is 
	 * 						distributed <i>X - location ~ Log-N(...)</i>.
	 * @param positiveDir	<code>True</code> if the distribution is defined from the 
	 * 						{@link # location} in the positive direction. <code>False</code> if it 
	 * 						is defined in the negative direction.
	 * @param x				The independent value
	 * @return The value or quantile
	 */
	public static double computeCDF(double miu, double sigma, double location, 
										boolean positiveDir, double x)
	{
		int direction	= positiveDir ? 1 : -1;
		if (direction*(x - location) < 0.0)
			return positiveDir ? 0.0 : 1.0;
		double xtrans	= Math.log(direction*(x - location));
		double normP	= Normal.computeCDF(miu, sigma, xtrans);
		return positiveDir ? normP : 1.0 - normP;
	}
	
	/**
	 * Computes the value of a random variable with an Log-normal distribution that has the 
	 * cumulative distribution function value that enters as a parameter. That is, the inverse 
	 * cumulative distribution function value. Assumes a standard location of 0 and a positive 
	 * direction. Returns {@link java.lang.Double#NaN} if value of the p parameter is smaller than 
	 * 0 or larger than 1.
	 * @param miu			The mean of the underlying normal distribution
	 * @param sigma			The standard deviation of the underlying normal distribution
	 * @param p				The value of the cumulative distribution function
	 * @return The value of a random variable with an Log-normal distribution that has the 
	 * cumulative distribution function value that enters as a parameter
	 */
	public static double computeInvCDF(double miu, double sigma, double p)
	{
		return computeInvCDF(miu, sigma, 0.0, true, p);
	}
	
	/**
	 * Computes the value of a random variable with an Log-normal distribution that has the 
	 * cumulative distribution function value that enters as a parameter. That is, the inverse 
	 * cumulative distribution function value. Returns {@link java.lang.Double#NaN} if value of the 
	 * p parameter is smaller than 0 or larger than 1.
	 * @param miu			The mean of the underlying normal distribution
	 * @param sigma			The standard deviation of the underlying normal distribution
	 * @param location		The origin of the Log-normal distribution. Usually it is set as zero. 
	 * 						However, it can be modified such that the random variable is 
	 * 						distributed <i>X - location ~ Log-N(...)</i>.
	 * @param positiveDir	<code>True</code> if the distribution is defined from the 
	 * 						{@link # location} in the positive direction. <code>False</code> if it 
	 * 						is defined in the negative direction.
	 * @param p				The value of the cumulative distribution function
	 * @return The value of a random variable with an Log-normal distribution that has the 
	 * cumulative distribution function value that enters as a parameter
	 */
	public static double computeInvCDF(double miu, double sigma, double location, 
										boolean positiveDir, double p)
	{
		if(p < 0.0 || p > 1.0)
			return Double.NaN;
		
		double modP		= positiveDir ? p : 1.0 - p;
		int direction	= positiveDir ? 1 : -1;
		double invNorm	= Normal.computeInvCDF(miu, sigma, modP);
		return location + direction*Math.exp(invNorm);
	}
	
	/**
	 * Computes the root mean square error from comparing the accumulated weight function of a 
	 * series of data points to the cumulative probability density function of a Log-normal 
	 * distribution. This indicators helps measure how well the distribution fits the data. Assumes 
	 * a standard location of 0 and a positive direction.
	 * @param miu			The mean of the underlying normal distribution
	 * @param sigma			The standard deviation of the underlying normal distribution
	 * @param values		The data points to be compared
	 * @return The root mean square error from comparing the accumulated weight function of a 
	 * series of data points to the cumulative probability density function of a Log-normal 
	 * distribution
	 */
	public static double computeRMSE(double miu, double sigma, ContSeries values)
	{
		return computeRMSE(miu, sigma, 0.0, true, values);
	}
	
	/**
	 * Computes the root mean square error from comparing the accumulated weight function of a 
	 * series of data points to the cumulative probability density function of a Log-normal 
	 * distribution. This indicators helps measure how well the distribution fits the data.
	 * @param miu			The mean of the underlying normal distribution
	 * @param sigma			The standard deviation of the underlying normal distribution
	 * @param location		The origin of the Log-normal distribution. Usually it is set as zero. 
	 * 						However, it can be modified such that the random variable is 
	 * 						distributed <i>X - location ~ Log-N(...)</i>.
	 * @param positiveDir	<code>True</code> if the distribution is defined from the 
	 * 						{@link # location} in the positive direction. <code>False</code> if it 
	 * 						is defined in the negative direction.
	 * @param values		The data points to be compared
	 * @return The root mean square error from comparing the accumulated weight function of a 
	 * series of data points to the cumulative probability density function of a Log-normal 
	 * distribution
	 */
	public static double computeRMSE(double miu, double sigma, double location, 
										boolean positiveDir, ContSeries values)
	{
		LogNormal logNormal	= new LogNormal(miu, sigma, location, positiveDir);
		return logNormal.computeRMSE(values);
	}
	
	/**
	 * Generates a random number with a given Log-normal distribution. Assumes a standard location 
	 * of 0 and a positive direction.
	 * @param miu			The mean of the underlying normal distribution
	 * @param sigma			The standard deviation of the underlying normal distribution
	 * @return A random number with a given Log-normal distribution
	 */
	public static double sample(double miu, double sigma)
	{
		return sample(miu, sigma, 0.0, true);
	}
	
	/**
	 * Generates a random number with a given Log-normal distribution
	 * @param miu			The mean of the underlying normal distribution
	 * @param sigma			The standard deviation of the underlying normal distribution
	 * @param location		The origin of the Log-normal distribution. Usually it is set as zero. 
	 * 						However, it can be modified such that the random variable is 
	 * 						distributed <i>X - location ~ Log-N(...)</i>.
	 * @param positiveDir	<code>True</code> if the distribution is defined from the 
	 * 						{@link # location} in the positive direction. <code>False</code> if it 
	 * 						is defined in the negative direction.
	 * @return A random number with a given Log-normal distribution
	 */
	public static double sample(double miu, double sigma, double location, 
										boolean positiveDir)
	{
		double standard	= Normal.sample(miu, sigma);
		int direction	= positiveDir ? 1 : -1;
		return location + direction*Math.exp(standard);
	}
	
}
