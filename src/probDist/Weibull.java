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

/**
 * This class defines static methods for making computations with Weibull probability 
 * distributions
 * Available methods include: <ul>
 * <li>Compute the probability of a range in a Weibull distribution
 * <li>Compute the expectation of a range of a Weibull distribution
 * <li>Compute the CDF and inverse CDF functions for a Weibull distribution
 * <li>Generate random numbers with a Weibull distribution </ul>
 * @author Felipe Hernández
 */
public class Weibull
{
	
	/**
	 * Computes the value of the cumulative distribution function of a Weibull probability 
	 * distribution. Returns Double.NaN if k is greater than 0.
	 * @param lambda The scale parameter of the Weibull distribution. If lambda is positive, the
	 * Weibull probability density function is defined in the positive direction. If lambda is 
	 * negative, the probability density function is reflected around the density axis.
	 * @param k The shape parameter of the Weibull distribution. Must be greater than 0. Use 1 for 
	 * an Exponential distribution.
	 * @param x The independent value
	 * @return The value or quantile
	 */
	public static double computeCDF(double lambda, double k, double x)
	{
		return computeCDF(lambda, k, 0, x);
	}

	/**
	 * Computes the value of the cumulative distribution function of a Weibull probability 
	 * distribution. Returns Double.NaN if k is greater than 0.
	 * @param lambda The scale parameter of the Weibull distribution. If lambda is positive, the
	 * Weibull probability density function is defined in the positive direction. If lambda is 
	 * negative, the probability density function is reflected around the density axis.
	 * @param k The shape parameter of the Weibull distribution. Must be greater than 0. Use 1 for 
	 * an Exponential distribution.
	 * @param location The center of the Weibull distribution. Usually it is set as 0. However, it 
	 * can be modified such that the random variable is distributed X - location ~ Wei(lambda, k).
	 * @param x The independent value
	 * @return The value or quantile
	 */
	public static double computeCDF(double lambda, double k, double location, double x)
	{
		x = x - location;
		if(k <= 0)
			return Double.NaN;
		if(lambda == 0.0)
			if(x < 0)
				return 0;
			else
				return 1;
		else if(lambda > 0)
			if(x < 0)
				return 0;
			else
				return 1.0 - Math.exp(-Math.pow(x/lambda,k));
		else
			if(x > 0)
				return 1;
			else
				return Math.exp(-Math.pow(x/lambda,k));
	}
	
	/**
	 * Computes the value of a random variable with a Weibull distribution that has the 
	 * cumulative distribution function value that enters as a parameter. That is, the inverse 
	 * cumulative distribution function value. Returns Double.NaN if the value of the p parameter 
	 * is smaller than 0 or larger than 1, or if k is not greater than 0.
	 * @param lambda The scale parameter of the Weibull distribution. If lambda is positive, the
	 * Weibull probability density function is defined in the positive direction. If lambda is 
	 * negative, the probability density function is reflected around the density axis.
	 * @param k The shape parameter of the Weibull distribution. Must be greater than 0. 1 for an 
	 * Exponential distribution.
	 * @param p The value of the cumulative distribution function
	 * @return The value of a random variable with a Weibull distribution that has the 
	 * cumulative distribution function value that enters as a parameter
	 */
	public static double computeInvCDF(double lambda, double k, double p) 
	{
		return computeInvCDF(lambda, k, 0, p);
	}
	
	/**
	 * Computes the value of a random variable with a Weibull distribution that has the 
	 * cumulative distribution function value that enters as a parameter. That is, the inverse 
	 * cumulative distribution function value. Returns Double.NaN if value of the p parameter is 
	 * smaller than 0 or larger than 1, or if k is not greater than 0.
	 * @param lambda The scale parameter of the Weibull distribution. If lambda is positive, the
	 * Weibull probability density function is defined in the positive direction. If lambda is 
	 * negative, the probability density function is reflected around the density axis.
	 * @param k The shape parameter of the Weibull distribution. Must be greater than 0. 1 for an 
	 * Exponential distribution.
	 * @param location The center of the Weibull distribution. Usually it is set as 0. However, it 
	 * can be modified such that the random variable is distributed X - location ~ Wei(lambda, k).
	 * @param p The value of the cumulative distribution function
	 * @return The value of a random variable with a Weibull distribution that has the 
	 * cumulative distribution function value that enters as a parameter
	 */
	public static double computeInvCDF(double lambda, double k, double location, double p) 
	{
		if(p < 0 || p > 1 || k <= 0)
			return Double.NaN;
		if(lambda == 0.0)
			return location;
		else if(lambda > 0)
			if(p == 1)
				return Double.POSITIVE_INFINITY;
			else
			{
				double power = 1.0/k;
			    return Math.pow(-Math.log(1.0 - p), power)*lambda + location;
			}
		else
			if(p == 0)
				return Double.NEGATIVE_INFINITY;
			else
			{
				double power = 1.0/k;
				return Math.pow(-Math.log(p), power)*lambda + location;
			}
	}
	
	/**
	 * Generates a random number with a given Weibull distribution
	 * @param lambda The scale parameter of the Weibull distribution. If lambda is positive, the
	 * Weibull probability density function is defined in the positive direction. If lambda is 
	 * negative, the probability density function is reflected around the density axis.
	 * @param k The shape parameter of the Weibull distribution. Must be greater than 0. 1 for an 
	 * Exponential distribution.
	 * @return A random number with a given Weibull distribution
	 */
	public static double sample(double lambda, double k)
	{
		return sample(lambda, k, 0);
	}
	
	/**
	 * Generates a random number with a given Weibull distribution
	 * @param lambda The scale parameter of the Weibull distribution. If lambda is positive, the
	 * Weibull probability density function is defined in the positive direction. If lambda is 
	 * negative, the probability density function is reflected around the density axis.
	 * @param k The shape parameter of the Weibull distribution. Must be greater than 0. 1 for an 
	 * Exponential distribution.
	 * @param location The center of the Weibull distribution. Usually it is set as 0. However, it 
	 * can be modified such that the random variable is distributed X - direction ~ Wei(lambda, k).
	 * @return A random number with a given Weibull distribution
	 */
	public static double sample(double lambda, double k, double location)
	{
		if(k <= 0)
			return Double.NaN;
		double random = Math.pow(-Math.log(1 - Math.random())*Math.abs(lambda), 1/k);
		return lambda > 0 ? random + location : -random + location;
	}
	
}
