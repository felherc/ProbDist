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
import java.util.Collections;

import probDist.Kernel;
import utilities.Utilities;
import utilities.geom.Point2D;
import utilities.stat.ContSeries;

/**
 * This class represents an univariate continuous probability distribution which is modeled
 * using kernel density estimation over a set of samples.
 * @author Felipe Hernández
 */
public class KernelDensity extends ContProbDist
{

	// --------------------------------------------------------------------------------------------
	// Constants
	// --------------------------------------------------------------------------------------------
	
	/**
	 * Kernel density distribution type String identifier
	 */
	public final static String ID = "Kernel density";
	
	/**
	 * Kernel density distribution type short String identifier
	 */
	public final static String SHORT_ID = "Ker";
	
	/**
	 * Default value for {@link #kernelFunction}
	 */
	public final static int DEF_KERNEL_TYPE = Kernel.KERNEL_GAUSSIAN;
	
	/**
	 * Default value for {@link #weighted}
	 */
	public final static boolean DEF_WEIGHTED = false;
	
	/**
	 * Default value for {@link #bandwidth}
	 */
	public final static double DEF_BANDWIDTH = 1.0;
	
	/**
	 * Default weight assigned to samples
	 */
	public final static double DEF_WEIGTH = 1.0;
	
	/**
	 * Acceptable precision for the error on the computation of the inverse CDF
	 */
	public final static double INV_CDF_PRECISION = 1E-7;
	
	/**
	 * Maximum exponent for the exponential search of the inverse CDF on the tails
	 */
	public final static double INV_CDF_MAX_EXPONENT = 20;
	
	// --------------------------------------------------------------------------------------------
	// Attributes
	// --------------------------------------------------------------------------------------------
	
	/**
	 * Identifier of the kernel function to use for the density estimation. The functions are 
	 * identified by the following constants:<ul>
	 * <li>{@link probDist.Kernel#KERNEL_UNIFORM}
	 * <li>{@link probDist.Kernel#KERNEL_TRIANGULAR}
	 * <li>{@link probDist.Kernel#KERNEL_EPANECHNIKOV}
	 * <li>{@link probDist.Kernel#KERNEL_QUARTIC}
	 * <li>{@link probDist.Kernel#KERNEL_TRIWEIGHT}
	 * <li>{@link probDist.Kernel#KERNEL_GAUSSIAN}
	 * <li>{@link probDist.Kernel#KERNEL_COSINE}
	 * <li>{@link probDist.Kernel#KERNEL_LOGISTIC}
	 * <li>{@link probDist.Kernel#KERNEL_SILVERMAN}
	 * </ul>
	 */
	private int kernelFunction;
	
	/**
	 * True if the samples have different weights. False if every sample has the same weight. That 
	 * is, if each sample has a kernel with the same contribution to the probability density.
	 */
	private boolean weighted;
	
	/**
	 * Smoothing parameter for the kernel density estimation. The bandwidth represents the extent 
	 * of the kernel function used. Higher values result in smoother densities. Should be positive.
	 */
	private double bandwidth;
	
	/**
	 * List of the samples to estimate the distribution from. The samples are represented as 
	 * instances of {@link utilities.geom.Point2D}, where <i>x</i> is the value of the sample and 
	 * <i>y</i> is its weight.
	 */
	private ArrayList<Point2D> samples;
	
	/**
	 * List with the values in the samples. Used for computing the mean, standard deviation, and 
	 * other statistics from the data.
	 */
	private ContSeries statistics;
	
	// --------------------------------------------------------------------------------------------
	// Constructors
	// --------------------------------------------------------------------------------------------
	
	/**
	 * Default constructor; defines default values for the parameters
	 */
	public KernelDensity()
	{
		type			= ContProbDist.KERNEL;
		kernelFunction	= DEF_KERNEL_TYPE;
		weighted		= DEF_WEIGHTED;
		bandwidth		= DEF_BANDWIDTH;
		samples			= new ArrayList<Point2D>();
		statistics		= new ContSeries(weighted);
	}
	
	// --------------------------------------------------------------------------------------------
	// Methods
	// --------------------------------------------------------------------------------------------

	/**
	 * @return {@link #kernelFunction}
	 */
	public int getKernelFunction() 
	{
		return kernelFunction;
	}

	/**
	 * @param kernelFunction {@link #kernelFunction}
	 */
	public void setKernelFunction(int kernelFunction) 
	{
		this.kernelFunction = kernelFunction;
	}

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
	public void setWeighted(boolean weighted) 
	{
		this.weighted = weighted;
		computeStatistics();
	}

	/**
	 * @return {@link #bandwidth}
	 */
	public double getBandwidth() 
	{
		return bandwidth;
	}

	/**
	 * @param bandwidth {@link #bandwidth}
	 */
	public void setBandwidth(double bandwidth) 
	{
		this.bandwidth = bandwidth;
	}
	
	/**
	 * @return A list with the values of the samples
	 */
	public ArrayList<Double> getSamples()
	{
		ArrayList<Double> list = new ArrayList<Double>();
		for (Point2D sample : samples)
			list.add(sample.x);
		return list;
	}

	/**
	 * @return {@link #samples}
	 */
	public ArrayList<Point2D> getSamplesWeights() 
	{
		return samples;
	}
	
	/**
	 * Clears {@link #samples}. Also clears {@link #statistics}.
	 */
	public void clearSamples()
	{
		samples.clear();
		statistics.clear();
	}
	
	/**
	 * Adds a new sample to {@link #samples} and updates {@link #statistics}. If the distribution
	 * is weighted, a default weight determined by {@link #DEF_WEIGTH} is used.
	 * @param value The value of the sample to add
	 */
	public void addSample(double value)
	{
		Point2D point = new Point2D(value, DEF_WEIGTH);
		samples.add(point);
		updateStatistics(point);
	}
	
	/**
	 * Adds a new sample to {@link #samples} and updates {@link #statistics}. If the distribution
	 * is not weighted, the weight parameter will not have any effect.
	 * @param value The value of the sample
	 * @param weight The weight of the sample
	 */
	public void addSample(double value, double weight)
	{
		weight 			= weight >= 0.0 ? weight : 0.0;
		Point2D point	= new Point2D(value, weight);
		samples.add(point);
		updateStatistics(point);
	}

	/**
	 * Sets {@link #samples} and computes {@link #statistics}
	 * <p>Time complexity: <i>O(number of samples)</i>
	 * <br>Space complexity: <i>O(number of samples)</i>
	 * @param samples {@link #samples}
	 */
	public void setSamples(ArrayList<Point2D> samples)
	{
		this.samples = samples;
		computeStatistics();
	}
	
	/**
	 * Clears {@link #statistics} and re-fills it with the data in {@link #samples}. Call this 
	 * method when the actual samples or their weights were modified externally.
	 * <p>Time complexity: <i>O(number of samples)</i>
	 * <br>Space complexity: <i>O(number of samples)</i>
	 */
	public void computeStatistics()
	{
		statistics = new ContSeries(weighted);
		for (Point2D sample : samples)
			updateStatistics(sample);
	}

	/**
	 * Updates {@link #statistics} with a new sample
	 * <p>Time complexity: <i>O(1)</i>
	 * <br>Space complexity: <i>O(1)</i>
	 * @param sample The sample to add
	 */
	private void updateStatistics(Point2D sample) 
	{
		double value	= sample.x;
		double weight	= sample.y;
		if (!Double.isNaN(value))
			if (weighted)
				statistics.addValue(value, weight);
			else
				statistics.addValue(value);
	}
	
	/**
	 * Estimates the bandwidth using the optimal criteria for an assumed underlying Gaussian 
	 * distributions as described in the reference below. The criteria is termed "normal 
	 * distribution approximation" or "Silverman's rule of thumb". <br><br>
	 * Silverman, B.W., 1998, <i>Density Estimation for Statistics and Data Analysis</i>, London: 
	 * Chapman & Hall/CRC, p.48, ISBN 0-412-24620-1.
	 * <p>Time complexity: <i>O(1)</i>
	 * <br>Space complexity: <i>O(1)</i>
	 */
	public void computeGaussianBandwidth()
	{
		double stDev	= statistics.getStDev();
		bandwidth		= Math.pow(4*Math.pow(stDev, 5)/(3*samples.size()), 0.2);
	}
	
	@Override
	public String getTypeString() 
	{
		return ID;
	}

	@Override
	public String toString()
	{
		String str = SHORT_ID + "(bw = " + bandwidth + ", {";
		for (Point2D sample : samples)
		{
			if (weighted)
				str	+= sample.y + ": " + sample.x + ", ";
			else
				str += sample.x + ", ";
		}
		Utilities.removeLast(str);
		str += "}";
		return str;
	}

	@Override
	public String toString(int decimalPlaces) 
	{
		String str = SHORT_ID + "(bw = " + Utilities.round(bandwidth) + ", {";
		for (Point2D sample : samples)
		{
			if (weighted)
				str	+= Utilities.round(sample.y) + ": " + Utilities.round(sample.x) + ", ";
			else
				str += Utilities.round(sample.x) + ", ";
		}
		Utilities.removeLast(str);
		str += "}";
		return str;
	}

	@Override
	public double getMean() 
	{
		return statistics.getMean();
	}

	@Override
	public double getStDev() 
	{
		double var = getVar();
		return Math.sqrt(var);
	}

	@Override
	public double getVar() 
	{
		if (samples.size() == 0)
			return Double.NaN;
		if (samples.size() == 1)
			return 0.0;
		
		double sum1			= 0;
		double sum2			= 0;
		double weightSum	= 0;
		for(Point2D sample : samples)
		{
			double compMean	= sample.x;
			double weight	= sample.y;
			sum1			+= weight*compMean;
			sum2			+= weight*compMean*compMean;
			weightSum		+= weight;
		}
		sum1				/= weightSum;
		sum2				/= weightSum;
		double kernelVar	= Kernel.getVar(kernelFunction);
		return kernelVar*bandwidth*bandwidth + sum2 - sum1*sum1;
	}
	
	@Override
	public double getSkewness()
	{
		// TODO Implement
		return Double.NaN;
	}
	
	@Override
	public ContProbDist truncate(double min, double max)
	{
		// TODO Implement
		return null;
	}

	/**
	 * Computes the probability density of the distribution evaluated at the provided value. The 
	 * density is computed using the kernel functions centered on the samples. The density 
	 * correspond's to the provided value's likelihood given the distribution.
	 * <p>Time complexity: <i>O(number of samples)</i>
	 * <br>Space complexity: <i>O(1)</i>
	 * @param value The value at which to evaluate the density
	 * @return The value of the probability density of the distribution evaluated at the provided 
	 * value
	 */
	public double getpdf(double x)
	{
		double sum = 0;
		double density;
		if (weighted)
		{
			for (Point2D sample : samples)
				sum += sample.y*Kernel.getpdf(x, sample.x, bandwidth, kernelFunction);
			density = sum/statistics.getWeightSum();
		}
		else
		{
			for (Point2D sample : samples)
				sum += Kernel.getpdf(x, sample.x, bandwidth, kernelFunction);
			density = sum/samples.size();
		}
		return density;
	}

	@Override
	public double getCDF(double x)
	{
		double sum = 0;
		double cdf;
		if (weighted)
		{
			for (Point2D sample : samples)
				sum += sample.y*Kernel.getCDF((x - sample.x)/bandwidth, kernelFunction);
			cdf = sum/statistics.getWeightSum();
		}
		else
		{
			for (Point2D sample : samples)
				sum += Kernel.getCDF((x - sample.x)/bandwidth, kernelFunction);
			cdf = sum/samples.size();
		}
		return cdf;
	}

	@Override
	public double getInvCDF(double p)
	{
		if (p > 1.0 || p < 0.0)
			return Double.NaN;
		
		// Determine if the value is on the tails of the distribution
		boolean onTails	= false;
		if (weighted)
		{
			double weightMin	= statistics.getValuesWeights().get(statistics.getIndexMin()).y;
			double weightMax	= statistics.getValuesWeights().get(statistics.getIndexMax()).y;
			double weightSum	= statistics.getWeightSum();
			onTails				= p < weightMin/(2*weightSum) || (1 - p) < weightMax/(2*weightSum);
		}
		else
			onTails	= p < 1.0/(2*samples.size()) || (1 - p) < 1.0/(2*samples.size());
		
		double invCDF	= Double.NaN;
		if (!onTails)
		{
			// Binary search within range
			double min		= statistics.getMin() - bandwidth;
			double max		= statistics.getMax() + bandwidth;
			double error	= Double.POSITIVE_INFINITY;
			while (Math.abs(error) > INV_CDF_PRECISION)
			{
				invCDF			= (max + min)/2;
				double tempP	= getCDF(invCDF);
				error			= p - tempP;
				if (error > 0)
					min 		= invCDF;
				else
					max			= invCDF;
			}
		}
		else // The value is on the tails of the distribution
		{
			boolean leftTail	= p < 0.5;
			double anchor		= leftTail 	? statistics.getMin() + bandwidth 
											: statistics.getMax() - bandwidth;
			double min			= 0;
			double max			= INV_CDF_MAX_EXPONENT;
			
			// Check for infinite values
			double tempP		= getCDF(anchor + bandwidth*Math.exp(max)*(leftTail ? -1.0 : 1.0));
			if (leftTail ? tempP > p : tempP < p)
				return leftTail ? Double.NEGATIVE_INFINITY : Double.POSITIVE_INFINITY;
			
			// Binary search on exponential space on tails
			double error		= Double.POSITIVE_INFINITY;
			while (Math.abs(error) > INV_CDF_PRECISION)
			{
				double exp	= (max + min)/2;
				invCDF		= anchor + bandwidth*Math.exp(exp)*(leftTail ? -1.0 : 1.0);
				tempP		= getCDF(invCDF);
				error		= p - tempP;
				if (leftTail ? error > 0 : error < 0)
					max		= exp;
				else
					min 	= exp;
			}
		}
		return invCDF;
	}
	
	/**
	 * Computes the ensemble version of the continuous ranked probability score (CRPS) of the
	 * samples given a reference value as described in the reference below. The CRPS measures how
	 * well the ensemble of samples approximates a target value that is considered to be the
	 * reality. Smaller values for the CRPS indicate that the samples are a better approximation
	 * to reality. The CRPS is considered a generalization of the mean absolute error (MAE) for
	 * probabilistic estimates and it collapses to the MAE in the case of a deterministic estimate.
	 * <br> J. Bröcker, “Evaluating raw ensembles with the continuous ranked probability score,”
	 * Q. J. R. Meteorol. Soc., vol. 138, no. 667, pp. 1611–1617, 2012.
	 * @param number The target value considered to be the reality
	 */
	public double computeEnsembleCRPS(double reference)
	{
		// Sort values
		ArrayList<Point2D> ordered	= new ArrayList<Point2D>();
		ordered.addAll(samples);
		Collections.sort(ordered);		// Sorts in ascending x order
		
		// Compute score
		double sum					= 0.0;
		for (int v = 0; v < ordered.size(); v++)
		{
			Point2D valueWeight		= ordered.get(v);
			double value			= valueWeight.getX();
			double nWeight			= valueWeight.getY()/statistics.getWeightSum();
			double alpha			= 0.0;
			for (int v2 = 0; v2 < v; v2++)
				alpha				+= ordered.get(v2).getY()/statistics.getWeightSum();
			alpha					-= nWeight/2;
			
			double diff				= reference - value;
			sum						+= nWeight*(diff >= 0.0 ? alpha*diff : -(1 - alpha)*diff);
		}
		return 2*sum;
	}

	/**
	 * Generates a random sample from this distribution. Returns {@link java.lang.Math#NaN} if
	 * there are no samples.
	 * <p>Time complexity: <i>O(number of samples)</i> if {@link #weighted};  <i>O(1)</i> otherwise
	 * <br>Space complexity: <i>O(1)</i>
	 * @return A random sample from this distribution
	 */
	public double sample()
	{
		if (samples.size() == 0)
			return Double.NaN;
		
		if (weighted)
		{
			double weightSum	= statistics.getWeightSum();
			if (weightSum <= 0)
				return Double.NaN;
			double selector		= Math.random()*weightSum;
			Point2D selected 	= null;
			double sum 			= 0;
			int index 			= 0;
			while (sum <= selector)
			{
				selected 		= samples.get(index);
				sum 			+= selected.y;
				index++;
			}
			return selected.x + bandwidth*Kernel.sample(kernelFunction);
		}
		else
		{
			int size			= samples.size();
			double selector		= Math.random()*size;
			int index			= (int)Math.floor(selector);
			Point2D selected 	= samples.get(index);
			return selected.x + bandwidth*Kernel.sample(kernelFunction);
		}
	}
	
	/**
	 * Generates a number of random sample from this distribution. Returns <i>null</i> if there are 
	 * no samples. A list of random roots (to indicate from which kernel to sample) is first 
	 * generated and then ordered. Then the samples in the distribution are visited in order to 
	 * perform the sampling from those indicated by the roots.
	 * <p>Time complexity: <i>O(n + c</i>*log<i>(c))</i> if weighted; <i>O(c)</i> otherwise
	 * <br>Space complexity: <i>O(c)</i>
	 * <br><i>n</i>: number of samples in the distribution; <i>c</i>: number of random samples to 
	 * generate
	 * @param sampleCount The number of random samples to generate
	 * @return A list with a number of random sample from this distribution
	 */
	public ArrayList<Double> sampleMultiple(int sampleCount)
	{
		if (samples.size() == 0)
			return null;
		
		ArrayList<Double> randSamp	= new ArrayList<>();
		if (!weighted)
		{
			for(int i = 0; i < sampleCount; i++)
				randSamp.add(sample());
			return randSamp;
		}
		
		// Create roots
		ArrayList<Double> roots		= new ArrayList<>();
		double weightSum			= statistics.getWeightSum();
		for(int i = 0; i < sampleCount; i++)
			roots.add(Math.random()*weightSum);
		Collections.sort(roots);
		
		// Create samples
		double sum 					= 0;
		int index					= -1;
		Point2D sample				= null;
		for (Double root : roots)
		{
			while (root >= sum || sample == null)
			{
				index++;
				sample				= samples.get(index);
				sum					+= sample.y;
			}
			randSamp.add(sample.x + bandwidth*Kernel.sample(kernelFunction));
		}
		Collections.shuffle(randSamp);
		return randSamp;
	}
	
	/**
	 * Generates a number of random sample from this distribution. Returns <i>null</i> if there are 
	 * no samples. Uses a different implementation than {@link #sampleMultiple} which might be
	 * faster for smaller number of random samples to generate: First the samples are assigned a
	 * value for the accumulated weight. Then, for each random sample to generate, a random root 
	 * (to indicate from which kernel to sample) is generated, and then a binary search is
	 * performed to find the adequate kernel to sample from.
	 * <p>Time complexity: <i>O(n + c</i>*log<i>(n))</i>
	 * <br>Space complexity: <i>O(n + c)</i>
	 * <br><i>n</i>: number of samples in the distribution; <i>c</i>: number of random samples to 
	 * generate
	 * @param sampleCount The number of random samples to generate
	 * @return A list with a number of random sample from this distribution
	 */
	public ArrayList<Double> sampleMultiple2(int sampleCount)
	{
		if (samples.size() == 0)
			return null;
		
		ArrayList<Double> randSamp	= new ArrayList<>();
		if (!weighted)
		{
			for(int i = 0; i < sampleCount; i++)
				randSamp.add(sample());
			return randSamp;
		}
		
		// Compute cumulative weights for samples
		ArrayList<Double> accWeights = new ArrayList<>();
		double sum = 0;
		for (Point2D sample : samples)
		{
			double weight = sample.y;
			accWeights.add(sum + weight);
			sum	+= weight;
		}
		
		// Perform binary search for each random sample
		double weightSum	= statistics.getWeightSum();
		for (int i = 0 ; i < sampleCount ; i++)
		{
			int min			= 0;
			int max			= samples.size() - 1;
			boolean ok		= false;
			double selector	= Math.random()*weightSum;
			int index		= 0;
			while (!ok)
			{
				double mid	= (double)(max + min)/2;
				index		= (int)(Math.random() < 0.5 ? Math.floor(mid) : Math.ceil(mid));
				sum			= accWeights.get(index);
				
				// Verify if the right sample was found
				if (index == 0)
					ok		= sum > selector;
				else
					ok		= sum > selector && accWeights.get(index - 1) <= selector;
					
				// Update limits
				if (sum < selector)
					min		= index;
				else
					max		= index;
			}
			randSamp.add(samples.get(index).x + bandwidth*Kernel.sample(kernelFunction));
		}
		Collections.shuffle(randSamp);
		return randSamp;
	}

	@Override
	public void shift(double newMean)
	{
		double displacement	= newMean - getMean();
		statistics.clear();
		for (Point2D sample : samples)
		{
			sample.x		+= displacement;
			updateStatistics(sample);
		}
	}

	@Override
	public void scale(double newStDev)
	{
		double newVar		= newStDev*newStDev;
		double oldVar		= getVar();
		double growRatio	= newVar/oldVar;
		bandwidth			= bandwidth*Math.sqrt(growRatio);
		double mean			= statistics.getMean();
		statistics.clear();
		for (Point2D sample : samples)
		{
			double diff		= sample.x - mean;
			double weight	= sample.y;
			double origSS	= diff*diff*weight;
			double newSS	= origSS*growRatio;
			sample.x		= mean + (diff > 0 ? 1 : -1)*Math.sqrt(newSS/weight);
			updateStatistics(sample);
		}
	}
	
	@Override
	public KernelDensity clone()
	{
		KernelDensity clone	= new KernelDensity();
		clone.setWeighted(weighted);
		clone.setKernelFunction(kernelFunction);
		clone.setProb(prob);
		for (Point2D sample : samples)
			clone.addSample(sample.x, sample.y);
		clone.setBandwidth(bandwidth);
		return clone;
	}
	
}
