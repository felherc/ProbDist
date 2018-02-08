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
 * This class allows computing the probability density for commonly used univariate kernel 
 * functions
 * @author Felipe Hernández
 */
public class Kernel 
{
	
	// --------------------------------------------------------------------------------------------
	// Constants
	// --------------------------------------------------------------------------------------------
	
	/**
	 * Identifier of the uniform kernel function
	 */
	public final static int KERNEL_UNIFORM = 0;
	
	/**
	 * Identifier of the triangular kernel function
	 */
	public final static int KERNEL_TRIANGULAR = 1;
	
	/**
	 * Identifier of the Epanechnikov kernel function
	 */
	public final static int KERNEL_EPANECHNIKOV = 2;
	
	/**
	 * Identifier of the quartic kernel function
	 */
	public final static int KERNEL_QUARTIC = 3;
	
	/**
	 * Identifier of the triweight kernel function
	 */
	public final static int KERNEL_TRIWEIGHT = 4;
	
	/**
	 * Identifier of the Gaussian kernel function
	 */
	public final static int KERNEL_GAUSSIAN = 5;
	
	/**
	 * Identifier of the cosine kernel function
	 */
	public final static int KERNEL_COSINE = 6;
	
	/**
	 * Identifier of the logistic kernel function
	 */
	public final static int KERNEL_LOGISTIC = 7;
	
	/**
	 * Identifier of the Silverman kernel function
	 */
	public final static int KERNEL_SILVERMAN = 8;

	// --------------------------------------------------------------------------------------------
	// Methods
	// --------------------------------------------------------------------------------------------
	
	/**
	 * Computes the standard deviation of the kernel
	 * @param kernelFunction The functions are determined by the following constants:<ul>
	 * <li>{@link #KERNEL_UNIFORM}
	 * <li>{@link #KERNEL_TRIANGULAR}
	 * <li>{@link #KERNEL_EPANECHNIKOV}
	 * <li>{@link #KERNEL_QUARTIC}
	 * <li>{@link #KERNEL_TRIWEIGHT}
	 * <li>{@link #KERNEL_GAUSSIAN}
	 * <li>{@link #KERNEL_COSINE}
	 * <li>{@link #KERNEL_LOGISTIC}
	 * <li>{@link #KERNEL_SILVERMAN}
	 * </ul>
	 * @return The variance of the kernel
	 */
	public static double getStDev(int kernelFunction) 
	{
		double var	= getVar(kernelFunction);
		return Math.sqrt(var);
	}
	
	/**
	 * Computes the variance of the kernel
	 * @param kernelFunction The functions are determined by the following constants:<ul>
	 * <li>{@link #KERNEL_UNIFORM}
	 * <li>{@link #KERNEL_TRIANGULAR}
	 * <li>{@link #KERNEL_EPANECHNIKOV}
	 * <li>{@link #KERNEL_QUARTIC}
	 * <li>{@link #KERNEL_TRIWEIGHT}
	 * <li>{@link #KERNEL_GAUSSIAN}
	 * <li>{@link #KERNEL_COSINE}
	 * <li>{@link #KERNEL_LOGISTIC}
	 * <li>{@link #KERNEL_SILVERMAN}
	 * </ul>
	 * @return The variance of the kernel
	 */
	public static double getVar(int kernelFunction) 
	{
		switch(kernelFunction)
		{
			case KERNEL_UNIFORM:		return 1.0/3;
			case KERNEL_GAUSSIAN:		return 1.0;
			default:					return Double.NaN;
			
			// TODO Implement the variance expressions for other kernels
		}
	}

	/**
	 * Computes the probability of the value of a kernel function to be within the provided 
	 * interval
	 * @param u1 The lower limit of the evaluation interval
	 * @param u2 The upper limit of the evaluation interval
	 * @param kernelFunction The functions are determined by the following constants:<ul>
	 * <li>{@link #KERNEL_UNIFORM}
	 * <li>{@link #KERNEL_TRIANGULAR}
	 * <li>{@link #KERNEL_EPANECHNIKOV}
	 * <li>{@link #KERNEL_QUARTIC}
	 * <li>{@link #KERNEL_TRIWEIGHT}
	 * <li>{@link #KERNEL_GAUSSIAN}
	 * <li>{@link #KERNEL_COSINE}
	 * <li>{@link #KERNEL_LOGISTIC}
	 * <li>{@link #KERNEL_SILVERMAN}
	 * </ul>
	 * @return The probability of the value of a kernel function to be within the provided 
	 * interval
	 */
	public static double getProb(double u1, double u2, int kernelFunction)
	{
		double cdf1 = getCDF(u1, kernelFunction);
		double cdf2 = getCDF(u2, kernelFunction);
		return cdf2 - cdf1;
	}
	
	/**
	 * Computes the unnormalized probability density of a specified kernel evaluated at 
	 * <b>u</b>
	 * @param u The value at which to evaluate the density
	 * @param kernelFunction The functions are determined by the following constants:<ul>
	 * <li>{@link #KERNEL_UNIFORM}
	 * <li>{@link #KERNEL_TRIANGULAR}
	 * <li>{@link #KERNEL_EPANECHNIKOV}
	 * <li>{@link #KERNEL_QUARTIC}
	 * <li>{@link #KERNEL_TRIWEIGHT}
	 * <li>{@link #KERNEL_GAUSSIAN}
	 * <li>{@link #KERNEL_COSINE}
	 * <li>{@link #KERNEL_LOGISTIC}
	 * <li>{@link #KERNEL_SILVERMAN}
	 * </ul>
	 * @return The unnormalized probability density of a specified kernel evaluated at 
	 * <b>u</b>. {@link java.lang.Double#NaN} if an invalid value for <b>kernelType</b> is 
	 * provided.
	 */
	public static double getDensity(double u, int kernelFunction)
	{
		switch(kernelFunction)
		{
			case KERNEL_UNIFORM:		return uniform(u);
			case KERNEL_TRIANGULAR:		return triangular(u);
			case KERNEL_EPANECHNIKOV:	return epanechnikov(u);
			case KERNEL_QUARTIC:		return quartic(u);
			case KERNEL_TRIWEIGHT:		return triweight(u);
			case KERNEL_GAUSSIAN:		return gaussian(u);
			case KERNEL_COSINE:			return cosine(u);
			case KERNEL_LOGISTIC:		return logistic(u);
			case KERNEL_SILVERMAN:		return silverman(u);
			default:					return Double.NaN;
		}
	}
	
	public static double getpdf(double x, double mean, double bandwidth, int kernelFunction)
	{
		switch(kernelFunction)
		{
			case KERNEL_UNIFORM:		return Uniform.computepdf(mean - bandwidth, 
					  											  mean + bandwidth, x);
			case KERNEL_GAUSSIAN:		return Normal.computepdf(mean, bandwidth, x);
			default:					return Double.NaN;
			
			// TODO Implement the normalized density for other kernel types
		}
	}
	
	/**
	 * Computes the cumulative distribution function of a specified kernel evaluated at <b>u</b>
	 * @param u The value at which to evaluate the density
	 * @param kernelFunction The functions are determined by the following constants:<ul>
	 * <li>{@link #KERNEL_UNIFORM}
	 * <li>{@link #KERNEL_TRIANGULAR}
	 * <li>{@link #KERNEL_EPANECHNIKOV}
	 * <li>{@link #KERNEL_QUARTIC}
	 * <li>{@link #KERNEL_TRIWEIGHT}
	 * <li>{@link #KERNEL_GAUSSIAN}
	 * <li>{@link #KERNEL_COSINE}
	 * <li>{@link #KERNEL_LOGISTIC}
	 * <li>{@link #KERNEL_SILVERMAN}
	 * </ul>
	 * @return The cumulative distribution function of a specified kernel evaluated at <b>u</b>.
	 * {@link java.lang.Double#NaN} if an invalid value for <b>kernelType</b> is provided. 
	 */
	public static double getCDF(double u, int kernelFunction)
	{
		switch(kernelFunction)
		{
			case KERNEL_UNIFORM:		return uniformCDF(u);
			case KERNEL_TRIANGULAR:		return triangularCDF(u);
			case KERNEL_EPANECHNIKOV:	return epanechnikovCDF(u);
			case KERNEL_QUARTIC:		return quarticCDF(u);
			case KERNEL_TRIWEIGHT:		return triweightCDF(u);
			case KERNEL_GAUSSIAN:		return gaussianCDF(u);
			case KERNEL_COSINE:			return cosineCDF(u);
			case KERNEL_LOGISTIC:		return logisticCDF(u);
			case KERNEL_SILVERMAN:		return silvermanCDF(u);
			default:					return Double.NaN;
		}
	}
	
	/**
	 * Generates a random sample from a specified kernel function
	 * @param kernelFunction The functions are determined by the following constants:<ul>
	 * <li>{@link #KERNEL_UNIFORM}
	 * <li>{@link #KERNEL_TRIANGULAR}
	 * <li>{@link #KERNEL_EPANECHNIKOV}
	 * <li>{@link #KERNEL_QUARTIC}
	 * <li>{@link #KERNEL_TRIWEIGHT}
	 * <li>{@link #KERNEL_GAUSSIAN}
	 * <li>{@link #KERNEL_COSINE}
	 * <li>{@link #KERNEL_LOGISTIC}
	 * <li>{@link #KERNEL_SILVERMAN}
	 * </ul>
	 * @return A random sample from a specified kernel function
	 * {@link java.lang.Double#NaN} if an invalid value for <b>kernelType</b> is provided. 
	 */
	public static double sample(int kernelFunction)
	{
		switch(kernelFunction)
		{
			case KERNEL_UNIFORM:		return uniformSample();
			case KERNEL_TRIANGULAR:		return triangularSample();
			case KERNEL_EPANECHNIKOV:	return epanechnikovSample();
			case KERNEL_QUARTIC:		return quarticSample();
			case KERNEL_TRIWEIGHT:		return triweightSample();
			case KERNEL_GAUSSIAN:		return gaussianSample();
			case KERNEL_COSINE:			return cosineSample();
			case KERNEL_LOGISTIC:		return logisticSample();
			case KERNEL_SILVERMAN:		return silvermanSample();
			default:					return Double.NaN;
		}
	}
	
	// --------------------------------------------------------------------------------------------
	// Density functions
	// --------------------------------------------------------------------------------------------
	
	/**
	 * Computes the probability density of a uniform kernel evaluated at <b>u</b>. The uniform
	 * kernel has an efficiency of 1.076 relative to the Epanechnikov kernel.
	 * @param u The value at which to evaluate the density
	 * @return The probability density of a uniform kernel evaluated at <b>u</b>
	 */
	public static double uniform(double u)
	{
		return Math.abs(u) > 1 ? 0 : 0.5;
	}
	
	/**
	 * Computes the probability density of a triangular kernel evaluated at <b>u</b>. The 
	 * triangular kernel has an efficiency of 1.014 relative to the Epanechnikov kernel.
	 * @param u The value at which to evaluate the density
	 * @return The probability density of a triangular kernel evaluated at <b>u</b>
	 */
	public static double triangular(double u)
	{
		double abs = Math.abs(u);
		return abs > 1 ? 0 : 1 - abs;
	}
	
	/**
	 * Computes the probability density of a uniform kernel evaluated at <b>u</b>. The Epanechnikov 
	 * kernel is optimal in a minimum variance sense,
	 * @param u The value at which to evaluate the density
	 * @return The probability density of an Epanechnikov kernel evaluated at <b>u</b>
	 */
	public static double epanechnikov(double u)
	{
		return Math.abs(u) > 1 ? 0 : 0.75*(1 - u*u);
	}
	
	/**
	 * Computes the probability density of a quartic kernel (or biweight kernel) evaluated at 
	 * <b>u</b>. The quartic kernel has an efficiency of 1.006 relative to the Epanechnikov kernel.
	 * @param u The value at which to evaluate the density
	 * @return The probability density of a quartic kernel evaluated at <b>u</b>
	 */
	public static double quartic(double u)
	{
		if (Math.abs(u) > 1)
			return 0;
		double temp = 1 - u*u;
		return (15/16)*temp*temp;
	}
	
	/**
	 * Computes the probability density of a triweight kernel evaluated at <b>u</b>. The triweight 
	 * kernel has an efficiency of 1.013 relative to the Epanechnikov kernel.
	 * @param u The value at which to evaluate the density
	 * @return The probability density of a triweight kernel evaluated at <b>u</b>
	 */
	public static double triweight(double u)
	{
		if (Math.abs(u) > 1)
			return 0;
		double temp = 1 - u*u;
		return (35/32)*temp*temp*temp;
	}
	
	/**
	 * Computes the probability density of a tricube kernel evaluated at <b>u</b>. The tricube 
	 * kernel has an efficiency of 1.002 relative to the Epanechnikov kernel.
	 * @param u The value at which to evaluate the density
	 * @return The probability density of a tricube kernel evaluated at <b>u</b>
	 */
	public static double tricube(double u)
	{
		double abs = Math.abs(u);
		if (abs > 1)
			return 0;
		double temp = 1 - abs*abs*abs;
		return (70/81)*temp*temp*temp;
	}
	
	/**
	 * Computes the probability density of a Gaussian kernel evaluated at <b>u</b>. The Gaussian 
	 * kernel has an efficiency of 1.051 relative to the Epanechnikov kernel.
	 * @param u The value at which to evaluate the density
	 * @return The probability density of a Gaussian kernel evaluated at <b>u</b>
	 */
	public static double gaussian(double u)
	{
		return Math.exp(-0.5*u*u)/(Math.sqrt(2*Math.PI));
	}
	
	/**
	 * Computes the probability density of a cosine kernel evaluated at <b>u</b>. The cosine 
	 * kernel has an efficiency of 1.0005 relative to the Epanechnikov kernel.
	 * @param u The value at which to evaluate the density
	 * @return The probability density of a cosine kernel evaluated at <b>u</b>
	 */
	public static double cosine(double u)
	{
		return Math.abs(u) > 1 ? 0 : (Math.PI/4)*Math.cos(Math.PI*u/2);
	}
	
	/**
	 * Computes the probability density of a logistic kernel evaluated at <b>u</b>. The logistic 
	 * kernel has an efficiency of 1.127 relative to the Epanechnikov kernel.
	 * @param u The value at which to evaluate the density
	 * @return The probability density of a logistic kernel evaluated at <b>u</b>
	 */
	public static double logistic(double u)
	{
		return 1/(Math.exp(u) + 2 + Math.exp(-u));
	}
	
	/**
	 * Computes the probability density of a Silverman kernel evaluated at <b>u</b>.
	 * @param u The value at which to evaluate the density
	 * @return The probability density of a Silverman kernel evaluated at <b>u</b>
	 */
	public static double silverman(double u)
	{
		double absU = Math.abs(u);
		double sqrt2 = Math.sqrt(2);
		return 0.5*Math.exp(-absU/sqrt2)*Math.sin(absU/sqrt2 + Math.PI/4);
	}
	
	// --------------------------------------------------------------------------------------------
	// Cumulative distribution functions
	// --------------------------------------------------------------------------------------------
	
	/**
	 * Computes the value of the cumulative distribution function of a uniform kernel evaluated at
	 * <b>u</b>
	 * @param u The value at which to evaluate the cumulative distribution function
	 * @return The value of the cumulative distribution function of a uniform kernel evaluated at
	 * <b>u</b>
	 */
	public static double uniformCDF(double u)
	{
		return Uniform.computeCDF(-1.0, 1.0, u);
	}
	
	/**
	 * Computes the value of the cumulative distribution function of a triangular kernel evaluated 
	 * at <b>u</b>
	 * @param u The value at which to evaluate the cumulative distribution function
	 * @return The value of the cumulative distribution function of a triangular kernel evaluated 
	 * at <b>u</b>
	 */
	public static double triangularCDF(double u)
	{
		if (u < 0.0)
			return u < 1.0 ? 0.0 : 0.5*(u	+ 1.0);
		else
			return u > 1.0 ? 0.0 : 0.5*(1.0	- u);
	}
	
	/**
	 * Computes the value of the cumulative distribution function of an Epanechnikov kernel 
	 * evaluated at <b>u</b>
	 * @param u The value at which to evaluate the cumulative distribution function
	 * @return The value of the cumulative distribution function of an Epanechnikov kernel 
	 * evaluated at <b>u</b>
	 */
	public static double epanechnikovCDF(double u)
	{
		if (u < -1) return 0.0;
		if (u >  1) return 1.0;
		return 0.75*(u - u*u*u/3 + 2/3);
	}
	
	/**
	 * Computes the value of the cumulative distribution function of a quartic kernel (or biweight 
	 * kernel) evaluated at <b>u</b>
	 * @param u The value at which to evaluate the cumulative distribution function
	 * @return The value of the cumulative distribution function of a quartic kernel (or biweight 
	 * kernel) evaluated at <b>u</b>
	 */
	public static double quarticCDF(double u)
	{
		if (u < -1) return 0.0;
		if (u >  1) return 1.0;
		return (15/16)*(u - 2*u*u/3 + Math.pow(u, 5)/5 + 8/15);
	}
	
	/**
	 * Computes the value of the cumulative distribution function of a triweight kernel evaluated 
	 * at <b>u</b>
	 * @param u The value at which to evaluate the cumulative distribution function
	 * @return The value of the cumulative distribution function of a triweight kernel evaluated 
	 * at <b>u</b>
	 */
	public static double triweightCDF(double u)
	{
		// TODO Implement
		return Double.NaN;
	}
	
	/**
	 * Computes the value of the cumulative distribution function of a tricube kernel evaluated 
	 * at <b>u</b>
	 * @param u The value at which to evaluate the cumulative distribution function
	 * @return The value of the cumulative distribution function of a tricube kernel evaluated 
	 * at <b>u</b>
	 */
	public static double tricubeCDF(double u)
	{
		// TODO Implement
		return Double.NaN;
	}
	
	/**
	 * Computes the value of the cumulative distribution function of a Gaussian kernel evaluated 
	 * at <b>u</b>
	 * @param u The value at which to evaluate the cumulative distribution function
	 * @return The value of the cumulative distribution function of a Gaussian kernel evaluated 
	 * at <b>u</b>
	 */
	public static double gaussianCDF(double u)
	{
		return Normal.computeCDF(u);
	}
	
	/**
	 * Computes the value of the cumulative distribution function of a cosine kernel evaluated 
	 * at <b>u</b>
	 * @param u The value at which to evaluate the cumulative distribution function
	 * @return The value of the cumulative distribution function of a cosine kernel evaluated 
	 * at <b>u</b>
	 */
	public static double cosineCDF(double u)
	{
		// TODO Implement
		return Double.NaN;
	}
	
	/**
	 * Computes the value of the cumulative distribution function of a logistice kernel evaluated 
	 * at <b>u</b>
	 * @param u The value at which to evaluate the cumulative distribution function
	 * @return The value of the cumulative distribution function of a logistic kernel evaluated 
	 * at <b>u</b>
	 */
	public static double logisticCDF(double u)
	{
		// TODO Implement
		return Double.NaN;
	}
	
	/**
	 * Computes the value of the cumulative distribution function of a Silverman kernel evaluated 
	 * at <b>u</b>
	 * @param u The value at which to evaluate the cumulative distribution function
	 * @return The value of the cumulative distribution function of a Silverman kernel evaluated 
	 * at <b>u</b>
	 */
	public static double silvermanCDF(double u)
	{
		// TODO Implement
		return Double.NaN;
	}
	
	// --------------------------------------------------------------------------------------------
	// Random sampling
	// --------------------------------------------------------------------------------------------
	
	/**
	 * @return A random sample from a uniform kernel
	 */
	public static double uniformSample()
	{
		return Uniform.sample(-1.0,  1.0);
	}
	
	/**
	 * @return A random sample from a triangular kernel
	 */
	public static double triangularSample()
	{
		// TODO Implement
		return Double.NaN;
	}
	
	/**
	 * @return A random sample from an Epanechnikov kernel
	 */
	public static double epanechnikovSample()
	{
		// TODO Implement
		return Double.NaN;
	}
	
	/**
	 * @return A random sample from a quartic kernel
	 */
	public static double quarticSample()
	{
		// TODO Implement
		return Double.NaN;
	}
	
	/**
	 * @return A random sample from a triweight kernel
	 */
	public static double triweightSample()
	{
		// TODO Implement
		return Double.NaN;
	}
	
	/**
	 * @return A random sample from a tricube kernel
	 */
	public static double tricubeSample()
	{
		// TODO Implement
		return Double.NaN;
	}
	
	/**
	 * @return A random sample from a Gaussian kernel
	 */
	public static double gaussianSample()
	{
		return Normal.sample(0.0, 1.0);
	}
	
	/**
	 * @return A random sample from a cosine kernel
	 */
	public static double cosineSample()
	{
		// TODO Implement
		return Double.NaN;
	}
	
	/**
	 * @return A random sample from a logistic kernel
	 */
	public static double logisticSample()
	{
		// TODO Implement
		return Double.NaN;
	}
	
	/**
	 * @return A random sample from a Silverman kernel
	 */
	public static double silvermanSample()
	{
		// TODO Implement
		return Double.NaN;
	}
	
}
