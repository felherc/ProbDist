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

import java.util.ArrayList;

import utilities.geom.Point2D;
import utilities.stat.ContSeries;

/**
 * This class represents probability distributions for continuous random variables
 * @author Felipe Hernández
 */
public abstract class ContProbDist
{
	
	// --------------------------------------------------------------------------------------------
	// Constants
	// --------------------------------------------------------------------------------------------
	
	/**
	 * Undefined distribution type identifier
	 */
	public final static int NONE = 0;
	
	/**
	 * Uniform distribution type identifier
	 */
	public final static int UNIFORM = 1;
	
	/**
	 * Normal distribution type identifier
	 */
	public final static int NORMAL = 2;
	
	/**
	 * Exponential distribution type identifier
	 */
	public final static int EXPONENTIAL = 3;
	
	/**
	 * Chi-square distribution type identifier
	 */
	public final static int CHI_SQUARE = 4;
	
	/**
	 * Gumbel distribution type identifier
	 */
	public final static int GUMBEL = 5;
	
	/**
	 * Log-normal distribution identifier
	 */
	public final static int LOG_NORMAL = 6;
	
	/**
	 * Weibull distribution type identifier
	 */
	public final static int WEIBULL = 7;
	
	/**
	 * Truncated normal distribution identifier
	 */
	public final static int TRUNC_NORMAL = 8;
	
	/**
	 * Kernel density distribution identifier
	 */
	public final static int KERNEL = 9;
	
	/**
	 * Mixture distribution type identifier
	 */
	public final static int MIXTURE = 10;
	
	/**
	 * Default number of points to compute the error between two distributions using the method
	 * {@link #computeRMSE(ContProbDist)}
	 */
	public final static int DEFAULT_SAMPLE_POINTS_ERROR = 50;
	
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
	// Constructors
	// --------------------------------------------------------------------------------------------
	
	/**
	 * Creates a continuous probability distribution from a series of values. The resulting
	 * distribution is selected from a series of available distributions whose cumulative 
	 * probability functions are compared to the data. The one with the least root mean square 
	 * error is selected. Only simple non-mixture distributions are tried. 
	 * @param values The series of values to create the distribution from
	 * @return The continuous probability distribution that fits the data. <code>null</code> if
	 * there are fewer than two values in the series.
	 */
	public static ContProbDist fromValues(ContSeries values)
	{
		return adjustProbDist(values);
	}
	
	/**
	 * Creates a continuous probability distribution from a series of values. The resulting
	 * distribution is selected from a series of available distributions whose cumulative 
	 * probability functions are compared to the data. The one with the least root mean square 
	 * error is selected. Only simple non-mixture distributions are tried. 
	 * @param values The series of values to create the distribution from
	 * @return The continuous probability distribution that fits the data. <code>null</code> if
	 * there are fewer than two values in the series.
	 */
	public static ContProbDist fromValues(ArrayList<Double> values)
	{
		ContSeries series = new ContSeries(false);
		for(Double value : values)
			series.addValue(value);
		return adjustProbDist(series);
	}
	
	// --------------------------------------------------------------------------------------------
	// Methods
	// --------------------------------------------------------------------------------------------
	
	/**
	 * Adjusts the continuous probability distribution to a series of values. The resulting
	 * distribution is selected from a series of available distributions whose cumulative 
	 * probability functions are compared to the data. The one with the least root mean square 
	 * error for the cumulative density curve is selected. Only simple non-mixture distributions 
	 * are tried.
	 * @param values The series of values to create the distribution from
	 * @return The continuous probability distribution that fits the data. <code>null</code> if
	 * there are fewer than two values in the series.
	 */
	public static ContProbDist adjustProbDist(ContSeries values)
	{
		int count					= values.size();
		if(count < 2)
			return null;
		else if(count == 2)
			return new Normal(values);
		else	// Compare distributions
		{
			Uniform uniform			= new Uniform(values);
			double uniformRMSE		= uniform.computeRMSE(values);
			
			Normal normal			= new Normal(values);
			double normalRMSE		= normal.computeRMSE(values);
			ContProbDist best		= uniformRMSE < normalRMSE ? uniform : normal;
			double minRMSE			= Math.min(uniformRMSE, normalRMSE);
			
			Exponential exponential	= new Exponential(values);
			double exponentialRMSE	= exponential.computeRMSE(values);
			best					= minRMSE < exponentialRMSE ? best : exponential;
			minRMSE					= Math.min(minRMSE, exponentialRMSE);
			
			ChiSquared chiSquared	= new ChiSquared(values);
			double chiSquaredRMSE	= chiSquared.computeRMSE(values);
			best					= minRMSE < chiSquaredRMSE ? best : chiSquared;
			minRMSE					= Math.min(minRMSE, chiSquaredRMSE);
			
			Gumbel gumbel			= new Gumbel(values);
			double gumbelRMSE		= gumbel.computeRMSE(values);
			best					= minRMSE < gumbelRMSE ? best : gumbel;
			minRMSE					= Math.min(minRMSE, gumbelRMSE);
			
			return best;
		}
	}
	
	/**
	 * @return The identifier of the type of distribution. The different types are defined as class
	 * constants
	 */
	public int getType()
	{
		return type;
	}
	
	/**
	 * @return The partial probability when used as a component in a mixture distribution
	 */
	public double getProb() 
	{
		return prob;
	}

	/**
	 * @param prob The partial probability when used as a component in a mixture distribution
	 */
	public void setProb(double prob) 
	{
		this.prob = prob < 0 ? 0 : prob;
	}
	
	/**
	 * Computes the probability of a random variable with this distribution of falling inside the 
	 * provided interval. Returns Double.NaN if the distribution has not been correctly defined.
	 * @param x1 The first limit of the sampling interval
	 * @param x2 The second limit of the sampling interval
	 * @return The probability of a random variable with this distribution of falling inside the 
	 * provided interval
	 */
	public double getProb(double x1, double x2)
	{
		return getCDF(x2) - getCDF(x1);
	}

	/**
	 * Computes the probability of a random variable with this distribution of being greater than 
	 * the provided value. Returns Double.NaN if the distribution has not been correctly defined.
	 * @param x The inferior limit of the sampling infinite interval
	 * @return The probability of a random variable with this distribution of being greater than 
	 * the provided value
	 */
	public double getSupProb(double x)
	{
		return getProb(x, Double.POSITIVE_INFINITY);
	}

	/**
	 * Computes the probability of a random variable with this distribution of being smaller than 
	 * the provided value. Returns Double.NaN if the distribution has not been correctly defined.
	 * @param x The superior limit of the sampling infinite interval
	 * @return The probability of a random variable with this distribution of being smaller than 
	 * the provided value
	 */
	public double getInfProb(double x)
	{
		return getProb(Double.NEGATIVE_INFINITY, x);
	}
	
	/**
	 * Computes the root mean square error from comparing the distribution to another continuous
	 * probability distribution. This indicator helps measure how this distribution resembles the
	 * other. The error is computed by partitioning the cumulative probability range into a number
	 * of samples given by {@link #DEFAULT_SAMPLE_POINTS_ERROR}.
	 * @param other The other continuous probability distribution to compare to
	 * @return The root mean square error from comparing the distribution to another continuous
	 * probability distribution
	 */
	public double computeRMSE(ContProbDist other)
	{
		return computeRMSE(other, DEFAULT_SAMPLE_POINTS_ERROR);
	}
	
	/**
	 * Computes the root mean square error from comparing the distribution to another continuous
	 * probability distribution. This indicator helps measure how this distribution resembles the
	 * other. The error is computed by partitioning the cumulative probability range into a number
	 * of samples.
	 * @param other		The other continuous probability distribution to compare to
	 * @param samples	The number of samples to partition the cumulative probability range
	 * @return The root mean square error from comparing the distribution to another continuous
	 * probability distribution
	 */
	public double computeRMSE(ContProbDist other, int samples)
	{
		double sum				= 0;
		samples					= samples < 2 ? 2 : samples;
		double delta			= ((double)samples/(samples + 1))/(samples - 1);
		for(double prob = delta/2 ; prob < 1.0 ; prob += delta)
		{
			double valueOther	= other.getInvCDF(prob);
			double cdfOther		= other.getCDF(valueOther);
			double error		= getCDF(valueOther) - cdfOther;
			sum					+= error*error;
		}		
		return Math.sqrt(sum/samples);
	}

	/**
	 * Computes the root mean square error from comparing the accumulated weight function of a 
	 * series of data points to the distribution's cumulative density function. This indicators 
	 * helps measure how well the distribution fits the data. Returns {@link java.lang.Double#NaN} 
	 * if the distribution has not been correctly defined.
	 * @param values The data points to be compared
	 * @return The root mean square error from comparing the accumulated weight function of a 
	 * series of data points to the distribution's cumulative density function
	 */
	public double computeRMSE(ContSeries values)
	{
		ArrayList<Point2D> points = values.getAcumWeights();
		double sum = 0;
		for(Point2D point : points)
		{
			double error = point.y - getCDF(point.x);
			sum += error*error;
		}
		return Math.sqrt(sum/points.size());
	}
	
	/**
	 * Computes the mean absolute relative error from comparing the distribution to another 
	 * continuous probability distribution. This indicator helps measure how this distribution 
	 * resembles the other. The error is computed by partitioning the cumulative probability range 
	 * into a number of samples given by {@link #DEFAULT_SAMPLE_POINTS_ERROR}.
	 * @param other	The other continuous probability distribution to compare to
	 * @return		The mean absolute relative error from comparing the distribution to another 
	 * 				continuous probability distribution
	 */
	public double computeMARE(ContProbDist other)
	{
		return computeMARE(other, DEFAULT_SAMPLE_POINTS_ERROR);
	}
	
	/**
	 * Computes the mean absolute relative error from comparing the distribution to another 
	 * continuous probability distribution. This indicator helps measure how this distribution 
	 * resembles the other. The error is computed by partitioning the cumulative probability range 
	 * into a number of samples.
	 * @param other		The other continuous probability distribution to compare to
	 * @param samples	The number of samples to partition the cumulative probability range
	 * @return			The mean absolute relative error from comparing the distribution to 
	 * 					another continuous probability distribution
	 */
	public double computeMARE(ContProbDist other, int samples)
	{
		samples					= samples < 2 ? 2 : samples;
		ContSeries series		= new ContSeries(false);
		double delta			= ((double)samples/(samples + 1))/(samples - 1);
		for(double prob = delta/2 ; prob < 1.0 ; prob += delta)
		{
			double valueOther	= other.getInvCDF(prob);
			double cdfOther		= other.getCDF(valueOther);
			double cdfThis		= getCDF(valueOther);
			series.addValue(Math.abs((cdfThis - cdfOther)/cdfThis));
		}
		return series.getMean();
	}
	
	/**
	 * Computes the mean absolute relative error from comparing the accumulated weight function of 
	 * a series of data points to the distribution's cumulative density function. This indicator 
	 * helps measure how well the distribution fits the data. Returns {@link java.lang.Double#NaN} 
	 * if the distribution has not been correctly defined.
	 * @param values	The data points to be compared
	 * @return 			The mean absolute relative error from comparing the accumulated weight 
	 * 					function of a series of data points to the distribution's cumulative 
	 * 					density function
	 */
	public double computeMARE(ContSeries values)
	{
		ContSeries series			= new ContSeries(false);
		ArrayList<Point2D> points	= values.getAcumWeights();
		for(Point2D point : points)
		{
			double cdf				=  getCDF(point.x);
			series.addValue(Math.abs((point.y - cdf)/point.y));
		}
		return series.getMean();
	}

	/**
	 * Returns the String identifier of the type of the distribution. The different types are 
	 * defined as class constants
	 * @return The String identifier of the type of distribution
	 */
	public abstract String getTypeString();
	
	/* (non-Javadoc)
	 * @see java.lang.Object#toString()
	 */
	public abstract String toString();
	
	/**
	 * Returns a String representation of the distribution with a given number of decimal places
	 * @param decimalPlaces The number of decimal places for the parameters of the distribution
	 * @return A String representation of the distribution with a given number of decimal places
	 */
	public abstract String toString(int decimalPlaces);
	
	/**
	 * @return The mean value of the distribution
	 */
	public abstract double getMean();

	/**
	 * @return The standard deviation of the distribution
	 */
	public abstract double getStDev();
	
	/**
	 * @return The variance of the distribution
	 */
	public abstract double getVar();
	
	/**
	 * @return The skewness of the distribution
	 */
	public abstract double getSkewness();
	
	/**
	 * Creates a truncated version of the probability distribution by limiting to the range between
	 * a minimum and a maximum value
	 * @param min	The lower bound of the resulting truncation range
	 * @param max	The upper bound of the resulting truncation range
	 * @return		The truncated distribution
	 */
	public abstract ContProbDist truncate(double min, double max);
	
	/**
	 * Computes the value of the probability density function at the provided value. Returns 
	 * {@link java.lang.Double#NaN} if the distribution has not been correctly defined.
	 * @param x The value at which to evaluate the probability density
	 * @return The value of the probability density function at the provided value
	 */
	public abstract double getpdf(double x);
	
	/**
	 * Computes the value of the cumulative distribution function of this probability distribution.
	 * Returns Double.NaN if the distribution has not been correctly defined.
	 * @param x The independent value
	 * @return The value or quantile
	 */
	public abstract double getCDF(double x);
	
	/**
	 * Computes the value of a random variable with this distribution that has the cumulative 
	 * distribution function value that enters as a parameter. That is, the inverse cumulative 
	 * distribution function value. Returns Double.NaN if value of the p parameter is smaller than 
	 * 0 or larger than 1, or if the distribution has not been correctly defined.
	 * @param p The value of the cumulative distribution function
	 * @return The value of a random variable with this distribution that has the cumulative 
	 * distribution function value that enters as a parameter
	 */
	public abstract double getInvCDF(double p);
	
	/**
	 * Generates a random sample from this distribution. Returns {@link java.lang.Double#NaN} if 
	 * the distribution has not been correctly defined.
	 * @return A random sample from this distribution
	 */
	public abstract double sample();
	
	/**
	 * Displaces the mean of the distribution to the value provided, maintaining its other
	 * attributes
	 * @param newMean The new value to center the distribution on 
	 */
	public abstract void shift(double newMean);
	
	/**
	 * Scales the distribution to match the standard deviation value provided, maintaining its
	 * other attributes
	 * @param newStDev The new standard deviation of the distribution
	 */
	public abstract void scale(double newStDev);
	
}