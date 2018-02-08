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
 * This class represents normal probability distributions and defines static methods for making 
 * computations with normal probability distributions. Available methods include: <ul>
 * <li>Compute the probability density of a value
 * <li>Compute the CDF and inverse CDF functions for a normal distribution
 * <li>Generate normally distributed random numbers 
 * <li>Compute the probability of a range in a normal distribution </ul>
 * @author Felipe Hernández
 */
public class Normal extends ContProbDist
{

	// --------------------------------------------------------------------------------------------
	// Constants
	// --------------------------------------------------------------------------------------------
	
	/**
	 * Normal distribution type String identifier
	 */
	public final static String ID = "Normal";
	
	/**
	 * Normal distribution type short String identifier
	 */
	public final static String SHORT_ID = "N";
	
	/**
	 * True if the polar form of the Box-Muller transformation function is used to generate random 
	 * numbers. False if the basic form is used.
	 */
	private final static boolean POLAR_NORMAL_GENERATOR = true;
	
	// --------------------------------------------------------------------------------------------
	// Attributes
	// --------------------------------------------------------------------------------------------
	
	/**
	 * The mean value
	 */
	private double mean;
	
	/**
	 * The standard deviation
	 */
	private double stDev;
	
	// --------------------------------------------------------------------------------------------
	// Constructors
	// --------------------------------------------------------------------------------------------
	
	/**
	 * Creates a normal probability distribution
	 * @param mean The mean value of the normal distribution
	 * @param stDev The standard deviation of the normal distribution
	 */
	public Normal(double mean, double stDev)
	{
		type = ContProbDist.NORMAL;
		this.mean = mean;
		this.stDev = stDev < 0 ? 0 : stDev;
	}
	
	/**
	 * Creates a normal probability distribution from a set of observed data. The distribution's
	 * parameters are set as the mean and standard deviation of the data
	 * @param values The data points to build the distribution from
	 */
	public Normal(ContSeries values)
	{
		type = ContProbDist.NORMAL;
		mean = values.getMean();
		stDev = values.getStDev();
	}
	
	// --------------------------------------------------------------------------------------------
	// Non-static methods
	// --------------------------------------------------------------------------------------------
	
	/**
	 * @return The mean value
	 */
	public double getMean()
	{
		return mean;
	}
	
	/**
	 * Sets the mean value of the normal distribution
	 * @param mean The value to set
	 */
	public void setMean(double mean)
	{
		this.mean = mean;
	}
	
	/**
	 * @return The standard deviation of the normal distribution
	 */
	public double getStDev()
	{
		return stDev;
	}
	
	/**
	 * Sets the standard deviation of the normal distribution
	 * @param stDev The value to set
	 */
	public void setStdDev(double stDev)
	{
		this.stDev = stDev;
	}
	
	@Override
	public double getVar() 
	{
		return stDev*stDev;
	}
	
	@Override
	public double getSkewness()
	{
		return 0.0;
	}
	
	@Override
	public String getTypeString() 
	{
		return ID;
	}

	@Override
	public String toString() 
	{
		return SHORT_ID + "(" + mean + ", " + stDev + ")";
	}
	
	@Override
	public String toString(int decimalPlaces) 
	{
		double roundedMean = Utilities.round(mean, decimalPlaces);
		double roundedStDev = Utilities.round(stDev, decimalPlaces);
		return SHORT_ID + "(" + roundedMean + ", " + roundedStDev + ")";
	}
	
	@Override
	public TruncatedNormal truncate(double min, double max)
	{
		return new TruncatedNormal(mean, stDev, min, max);
	}
	
	@Override
	public double getpdf(double x) 
	{
		return computepdf(mean, stDev, x);
	}

	@Override
	public double getCDF(double x) 
	{
		return computeCDF(mean, stDev, x);
	}

	@Override
	public double getInvCDF(double p) 
	{
		return computeInvCDF(mean, stDev, p);
	}

	@Override
	public double sample() 
	{
		return sample(mean, stDev);
	}
	
	@Override
	public void shift(double newMean)
	{
		setMean(newMean);
	}

	@Override
	public void scale(double newStDev)
	{
		setStdDev(newStDev);
	}
	
	// --------------------------------------------------------------------------------------------
	// Static methods
	// --------------------------------------------------------------------------------------------
	
	/**
	 * Computes the value of the probability density function of a normal probability 
	 * distribution
	 * @param mean The mean value of the normal distribution
	 * @param stDev The standard deviation of the normal distribution
	 * @param x The value at which to compute the probability density
	 * @return The value of the probability density function of a normal probability 
	 * distribution
	 */
	public static double computepdf(double mean, double stDev, double x)
	{
		if (stDev == 0.0)
			return x == mean ? Double.POSITIVE_INFINITY : 0.0;
		
		double dev	= x - mean;
		double exp	= -dev*dev/(2*stDev*stDev);
		return Math.exp(exp)/(stDev*Math.sqrt(2*Math.PI));
	}
	
	/**
	 * Computes the value of the cumulative distribution function of a normal probability 
	 * distribution
	 * @param mean The mean value of the normal distribution
	 * @param stDev The standard deviation of the normal distribution
	 * @param x The value to sample the CDF
	 * @return The value of the cumulative distribution function of a normal probability 
	 * distribution
	 */
	public static double computeCDF(double mean, double stDev, double x)
	{
		double z = (x - mean)/stDev;
		return computeCDF(z);
	}
	
	/**
	 * Computes the value of the cumulative distribution function of an standard normal probability 
	 * distribution
	 * @param z The value or quantile
	 */
	public static double computeCDF(double z)
	{
		double p;
		double zabs = Math.abs(z);
		
		if(zabs > 8.0)
			p = 0;
		else
		{
			final double root2pi = Math.sqrt(2*Math.PI);
			double expntl = Math.exp(-0.5*zabs*zabs);
		    double pdf = expntl/root2pi;
		    
		    if(zabs < 7.071)
		    {
		    	double p0 = 220.2068679123761;
			    double p1 = 221.2135961699311;
			    double p2 = 112.0792914978709;
			    double p3 = 33.91286607838300;
			    double p4 = 6.373962203531650;
			    double p5 = 0.7003830644436881;
			    double p6 = 0.3526249659989109E-1;
			    
			    double q0 = 440.4137358247522;
			    double q1 = 793.8265125199484;
			    double q2 = 637.3336333788311;
			    double q3 = 296.5642487796737;
			    double q4 = 86.78073220294608;
			    double q5 = 16.06417757920695;
			    double q6 = 1.755667163182642;
			    double q7 = 0.8838834764831844E-1;
		    	
		    	p = expntl*((((((p6*zabs + p5)*zabs + p4)*zabs + p3)*zabs + p2)*zabs + p1)*zabs + 
		    		p0)/(((((((q7*zabs + q6)*zabs + q5)*zabs + q4)*zabs + q3)*zabs + q2)*zabs + 
		    		q1)*zabs + q0);
		    }
		    else
		    	p = pdf/(zabs + 1.0/(zabs + 2.0/(zabs + 3.0/(zabs + 4.0/(zabs + 0.65)))));
		}
		
		return z < 0 ? p : 1.0 - p;
	}
	
	/**
	 * Computes the value of a random variable with a normal distribution that has the cumulative 
	 * distribution function value that enters as a parameter. That is, the inverse cumulative 
	 * distribution function value. Returns <code>Double.NaN</code> if the parameter value is 
	 * smaller than 0 or larger than 1.
	 * @param mean The mean value of the normal distribution
	 * @param stDev The standard deviation of the normal distribution
	 * @param p The value of the cumulative distribution function
	 * @return The value of a random variable with a normal distribution that has the cumulative 
	 * distribution function value that enters as a parameter
	 */
	public static double computeInvCDF(double mean, double stDev, double p)
	{
		if(p < 0 || p > 1)
			return Double.NaN;
		
		return (computeInvCDF(p)*stDev) + mean;
	}
	
	/**
	 * Computes the value of a random variable with an standard normal distribution that has the 
	 * cumulative distribution function value that enters as a parameter. That is, the inverse 
	 * cumulative distribution function value. Returns Double.NaN if the parameter value is smaller
	 * than 0 or larger than 1.
	 * @param p The value of the cumulative distribution function
	 * @return The value of a random variable with an standard normal distribution that has the 
	 * cumulative distribution function value that enters as a parameter
	 */
	public static double computeInvCDF(double p) 
	{
		if(p < 0 || p > 1)
			return Double.NaN;
		
		if(p == 0)
			return Double.NEGATIVE_INFINITY;
		
		if(p == 1.0)
			return Double.POSITIVE_INFINITY;
		
		double arg,t,t2,t3,xnum,xden,qinvp,x,pc;
		double c[] = {2.515517, 0.802853, 0.010328};
		double d[] = {1.432788, 0.189269, 0.001308};
		
		pc = p <= 0.5 ? p : 1 - p;
		arg = -2.0*Math.log(pc);
	    t = Math.sqrt(arg);
	    t2 = t*t;
	    t3 = t2*t;
	       
	    xnum = c[0] + c[1]*t + c[2]*t2;
	    xden = 1.0 + d[0]*t + d[1]*t2 + d[2]*t3;
	    qinvp = t - xnum/xden;
	    x = p <= 0.5 ? -qinvp : qinvp;
		
		return x;
	}
	
	/**
	 * Computes the root mean square error from comparing the accumulated weight function of a 
	 * series of data points to the cumulative probability density function of a normal 
	 * distribution. This indicators helps measure how well the uniform distribution fits the data.
	 * @param mean The mean value of the normal distribution
	 * @param stDev The standard deviation of the normal distribution
	 * @param values The data points to be compared
	 * @return The root mean square error from comparing the accumulated weight function of a 
	 * series of data points to the cumulative probability density function of a normal 
	 * distribution
	 */
	public static double computeRMSE(double mean, double stDev, ContSeries values)
	{
		Normal normal = new Normal(mean, stDev);
		return normal.computeRMSE(values);
	}
	
	/**
	 * Generates a normally distributed random number
	 * @param mean The mean value of the normal distribution
	 * @param stDev The standard deviation of the normal distribution
	 * @return A normally distributed random number
	 */
	public static double sample(double mean, double stDev)
	{
		if(POLAR_NORMAL_GENERATOR)
			return samplePolar(mean, stDev);
		else
			return sampleBasic(mean, stDev);
	}
	
	/**
	 * Generates a normally distributed random number using a specific method
	 * @param mean The mean value of the normal distribution
	 * @param stDev The standard deviation of the normal distribution
	 * @param polarMethod True if the polar form of the Box-Muller transformation function is used. 
	 * False if the basic form is used.
	 * @return A normally distributed random number
	 */
	public static double sample(double mean, double stDev, boolean polarMethod)
	{
		if(polarMethod)
			return samplePolar(mean, stDev);
		else
			return sampleBasic(mean, stDev);
	}
	
	/**
	 * Generates a normally distributed random number using the basic form of the Box-Muller 
	 * transform method
	 * @param mean The mean value of the normal distribution
	 * @param stDev The standard deviation of the normal distribution
	 * @return A normally distributed random number
	 */
	private static double sampleBasic(double mean, double stDev)
	{
		double random1 = 1 - Math.random();
		double random2 = 1 - Math.random();
		
		return stDev*Math.sqrt(-2*Math.log(random1))*
										Math.cos(2*Math.PI*random2) + mean;
	}
	
	/**
	 * Generates a normally distributed random number using the polar form of the Box-Muller 
	 * transform method
	 * @param mean The mean value of the normal distribution
	 * @param stDev The standard deviation of the normal distribution
	 * @return A normally distributed random number
	 */
	private static double samplePolar(double mean, double stDev)
	{
		double s = 0;
		double random1 = 0;
		double random2 = 0;
		
		while(s <= 0 || s >= 1)
		{
			random1 = 2*Math.random() - 1;
			random2 = 2*Math.random() - 1;
			s = Math.pow(random1, 2) + Math.pow(random2, 2);
		}
		
		return stDev*random1*Math.sqrt((-2*Math.log(s))/s) + mean;
	}
	
}