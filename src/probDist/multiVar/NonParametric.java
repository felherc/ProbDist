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

package probDist.multiVar;

import java.util.ArrayList;

import probDist.multiVar.tools.ContMultiSample;

public abstract class NonParametric extends MultiContProbDist
{

	// --------------------------------------------------------------------------------------------
	// Attributes
	// --------------------------------------------------------------------------------------------
	
	/**
	 * True if the samples have different weights. False if every sample has the same weight. That 
	 * is, if each sample has a kernel with the same contribution to the probability density.
	 */
	protected boolean weighted;
	
	/**
	 * List of the samples to estimate the distribution from
	 */
	protected ArrayList<ContMultiSample> samples;
	
	// --------------------------------------------------------------------------------------------
	// Methods
	// --------------------------------------------------------------------------------------------
	
	/**
	 * @return {@link #weighted}
	 */
	public boolean isWeighted() 
	{
		return weighted;
	}
	
	/**
	 * @param weighted {@link #weighted}
	 */
	public abstract void setWeighted(boolean weighted);
	
	/**
	 * @return {@link #samples}
	 */
	public ArrayList<ContMultiSample> getSamples()
	{
		return samples;
	}
	
	/**
	 * Returns a subset of the {@link samples} ordered by their weight. For unordered cases, or 
	 * when several samples have equal weights, the order is determined randomly between them.
	 * <p>Time complexity: <i>O(k)</i>
	 * <br>Space complexity: <i>O(1)</i>
	 * @param weightyFirst	True if the samples with the highest weight should be added first. 
	 * 						False if the ones with the lowest weight should be added first.
	 * @param weightPercent	The percentage of the total weight to be returned
	 * @param maximum		The maximum number of samples to be returned
	 * @return A list with the samples selected
	 */
	public abstract ArrayList<ContMultiSample> getSamples(boolean weightyFirst, 
															double weightPercent, int maximum);
	
	/**
	 * Clears {@link #samples}
	 */
	public abstract void clearSamples();
	
	/**
	 * Adds a new sample to {@link #samples}
	 * @param sample The sample to add
	 */
	public abstract void addSample(ContMultiSample sample);
	
	/**
	 * @param samples {@link #samples}
	 */
	public abstract void setSamples(ArrayList<ContMultiSample> samples);
	
	/**
	 * Computes the mean Mahalanobis distance between a provided point and the samples. The 
	 * distance to each sample is weighted if the distribution is.
	 * @param x	The point to compute the distance from
	 * @return The mean Mahalanobis distance between a provided point and the samples
	 */
	public abstract double getMeanMahalanobisDistance(double[] x);
	
	/**
	 * @param x	The point to compute the distance from
	 * @return	The Mahalanobis distance between the provided point and each of the samples that
	 * 			define the distribution
	 */
	public abstract double[] getMahalanobisDistanceToSamples(double[] x);
	
	/**
	 * Computes the mean Mahalanobis force between a provided point and the samples. The force is
	 * the inverse squared Mahalanobis distance to each sample, and the mean is weighted if the 
	 * distribution is.
	 * @param x	The point to compute the distance from
	 * @return The mean Mahalanobis force between a provided point and the samples
	 */
	public abstract double getMeanMahalanobisForce(double[] x);
	
	/**
	 * Computes the Mahalanobis force at the provided point. The force is defined here as the
	 * inverse of the mean inverse-weighted squared Mahalanobis distance between the point and the
	 * samples.
	 * @param x	The point to evaluate the force at
	 * @return	The Mahalanobis force at the provided point
	 */
	public abstract double getMahalanobisForce(double[] x);
	
	/**
	 * Computes the average independent standardized probability density function value at the 
	 * provided point. That is, first marginalizes the distribution for each dimension, then 
	 * computes the standardized pdf value of each marginalized distribution, and then computes
	 * the average of the individual independent values. The standardized pdf is the usual pdf
	 * multiplied by the distribution's standard deviation. This is analogous to first 
	 * standardizing the samples and then computing the pdf, and is done so that the pdf values
	 * of the marginal distributions are comparable between themselves independently of their 
	 * spread.
	 * @param x	The point at which to evaluate the average independent standardized 
	 * 			probability density function
	 * @return	The average independent standardized probability density function value
	 */
	public abstract double getMeanIndeppdf(double[] x);
	
}
