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
import java.util.Collections;
import java.util.HashSet;
import java.util.Hashtable;

import probDist.ContProbDist;
import probDist.Normal;
import probDist.multiVar.tools.ContMultiSample;
import probDist.multiVar.tools.GGMLiteCreator;
import probDist.multiVar.tools.Sample;
import utilities.Utilities;
import utilities.geom.PointID;
import utilities.stat.ContStats;

/**
 * This class represents kernel density multivariate normal (Gaussian) probability distributions 
 * which are parameterized by a set of (weighted) samples and a bandwidth (covariance) matrix. The 
 * covariance matrix is represented by a series of interconnected cliques of lower rank to improve 
 * storage and inference efficiency. Due to their sparse representation, they fall under the 
 * category of Gaussian Graphical Models (GGMs) and, due to their resource frugality, they earn the 
 * modifier "lite."
 * @author Felipe Hernández
 */
public class KD_GGMLite extends NonParametric
{

	// --------------------------------------------------------------------------------------------
	// Constants
	// --------------------------------------------------------------------------------------------
	
	/**
	 * Distribution type identifier
	 */
	public final static String ID = "Lightweight kernel density Gaussian graphical model";
	
	/**
	 * Error message: No samples in the distribution
	 */
	public static final String ERR_NO_SAMPLES = "No samples have been defined";
	
	/**
	 * Error message: Particle of wrong size
	 */
	public static final String ERR_PARTICLE_SIZE = "The particle does not have the same number "
								+ "of dimensions (%1$d) as the samples in the distribution (%2$d)";
	
	/**
	 * Error message: an offered sample contains a {@link java.lang.Double#NaN} value
	 */
	public static final String ERR_VALUE_IS_NAN = "The value on index %1$d of the offered sample "
													+ "is not a number (NaN)";
	
	// --------------------------------------------------------------------------------------------
	// Attributes
	// --------------------------------------------------------------------------------------------
	
	/**
	 * List with the values in the samples organized for each dimension. Used for computing the
	 * means, standard deviations, and other statistics from the data.
	 */
	private ArrayList<ContStats> statistics;
	
	/**
	 * A lightweight Gaussian graphical model that represents the kernels of the distribution. The
	 * covariance of the model should be scaled-down by {@link #scaling} to represent the kernels
	 * with their intended spread. 
	 */
	private GGMLite kernel;
	
	/**
	 * The scaling factor for the covariance matrix of the {@link #kernel}
	 */
	private double scaling;
	
	// --------------------------------------------------------------------------------------------
	// Constructors
	// --------------------------------------------------------------------------------------------
	
	public KD_GGMLite()
	{
		type				= MultiContProbDist.KERNEL_GGMLITE;
		weighted			= false;
		samples				= new ArrayList<>();
		statistics			= new ArrayList<>();
		kernel				= null;
	}
	
	/**	
	 * @param weighted {@link #weighted}
	 */
	public KD_GGMLite(boolean weighted)
	{
		type				= MultiContProbDist.KERNEL_GGMLITE;
		this.weighted		= weighted;
		samples				= new ArrayList<>();
		statistics			= new ArrayList<>();
		kernel				= null;
	}

	// --------------------------------------------------------------------------------------------
	// Methods - Getters and setters
	// --------------------------------------------------------------------------------------------
	
	@Override
	public String getTypeString()
	{
		return ID;
	}

	@Override
	public int getDimensionality()
	{
		return statistics.size();
	}
	
	/**
	 * @return {@link #weighted}
	 */
	public boolean isWeighted() 
	{
		return weighted;
	}
	
	/**
	 * Updates {@link #weighted} and computes {@link #statistics}
	 * @param weighted {@link #weighted}
	 * <p>Time complexity: <i>O(n*(d + k))</i>
	 * <br>Space complexity: <i>O(n*(d + k))</i>
	 * <br><i>n</i>: number of samples; <i>d</i>: number of dimensions; <i>k</i>: complexity
	 * of retrieving the weight and values from the {@link ContMultiSample} implementation
	 */
	public void setWeighted(boolean weighted) 
	{
		this.weighted = weighted;
		computeStatistics();
	}
	
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
	public ArrayList<ContMultiSample> getSamples(boolean weightyFirst, double weightPercent,
													int maximum)
	{
		ArrayList<ContMultiSample> result	= new ArrayList<>();
		int k								= samples.size();
		double weightSum					= 0.0;
		double totalWeight					= getWeightSum();
		if (weighted)
		{
			// Sort samples
			ArrayList<PointID> selector		= new ArrayList<>();
			for (int s = 0; s < samples.size(); s++)
			{
				ContMultiSample sample		= samples.get(s);
				selector.add(new PointID(s, sample.getWeight(), false));
			}
			Collections.shuffle(selector);
			Collections.sort(selector);
			
			// Select samples
			int s							= 0;
			while (result.size() < maximum && weightSum < weightPercent && s < k)
			{
				int index					= selector.get(s).getX();
				ContMultiSample sample		= samples.get(weightyFirst ? k - 1 - index : index);
				result.add(sample);
				s++;
				weightSum					+= sample.getWeight()/totalWeight;
			}
		}
		else
		{
			// Sort samples
			ArrayList<ContMultiSample> selector = new ArrayList<>();
			for (ContMultiSample sample : samples)
				selector.add(sample);
			Collections.shuffle(selector);
			
			// Select samples
			int s							= 0;
			while (result.size() < maximum && weightSum < weightPercent && s < k)
			{
				result.add(selector.get(s));
				s++;
				weightSum					= (double)s/k;
			}
		}
		return result;
	}
	
	/**
	 * Clears {@link #samples}. Also clears {@link #statistics}.
	 */
	public void clearSamples()
	{
		samples.clear();
		statistics.clear();
		kernel = null;
	}
	
	/**
	 * Adds a new sample to {@link #samples} and updates {@link #statistics}
	 * <p>Time complexity: <i>O(d + k)</i>
	 * <br>Space complexity: <i>O(d + k)</i>
	 * <br><i>d</i>: number of dimensions; <i>k</i>: complexity of retrieving the weight and values 
	 * from the {@link ContMultiSample} implementation
	 * @param sample The sample to add
	 */
	public void addSample(ContMultiSample sample)
	{
		for (int v = 0; v < sample.getValues().size(); v++)
			if (Double.isNaN(sample.getValues().get(v)))
				throw new IllegalArgumentException(String.format(ERR_VALUE_IS_NAN, v));		
		samples.add(sample);
		updateStatistics(sample);
	}
	
	/**
	 * Sets {@link #samples} and computes {@link #statistics}
	 * <p>Time complexity: <i>O(n*(d + k))</i>
	 * <br>Space complexity: <i>O(n*(d + k))</i>
	 * <br><i>n</i>: number of samples; <i>d</i>: number of dimensions; <i>k</i>: complexity
	 * of retrieving the weight and values from the {@link ContMultiSample} implementation
	 * @param samples {@link #samples}
	 */
	public void setSamples(ArrayList<ContMultiSample> samples)
	{
		this.samples = samples;
		computeStatistics();
	}
	
	/**
	 * @return The sum of the weights of all the samples
	 */
	public double getWeightSum()
	{
		return statistics.get(0).getWeightSum();
	}
	
	/**
	 * @return {@link #kernel}
	 */
	public GGMLite getKernel()
	{
		return kernel;
	}
	
	// --------------------------------------------------------------------------------------------
	// Methods - Computations
	// --------------------------------------------------------------------------------------------
	
	/**
	 * Clears {@link #statistics} and re-fills it with the data in {@link #samples}. Call this 
	 * method when the actual samples or their weights have been modified externally.
	 * <p>Time complexity: <i>O(n*(d + k))</i>
	 * <br>Space complexity: <i>O(n*(d + k))</i>
	 * <br><i>n</i>: number of samples; <i>d</i>: number of dimensions; <i>k</i>: complexity
	 * of retrieving the weight and values from the {@link ContMultiSample} implementation
	 */
	public void computeStatistics()
	{
		statistics.clear();
		for (ContMultiSample sample : samples)
			updateStatistics(sample);
	}

	/**
	 * Updates {@link #statistics} with a new sample
	 * <p>Time complexity: <i>O(d + k)</i>
	 * <br>Space complexity: <i>O(d + k)</i>
	 * <br><i>d</i>: number of dimensions; <i>k</i>: complexity of retrieving the weight and values 
	 * from the {@link ContMultiSample} implementation
	 * @param sample The sample to add
	 */
	private void updateStatistics(ContMultiSample sample) 
	{
		ArrayList<Double> values	= sample.getValues();
		int statsSize 				= statistics.size();
		int sampleSize				= values.size();
		if (statsSize < sampleSize)
			for (int size = statsSize + 1; size <= sampleSize; size++)
				statistics.add(new ContStats(weighted));
		double weight	= sample.getWeight();
		for (int i = 0; i < sampleSize; i++)
		{
			double value	= values.get(i);
			if (!Double.isNaN(value))
				statistics.get(i).addValue(value, weight);
		}
		kernel						= null;
	}
	
	/**
	 * @return A list with the marginal mean values of the distribution for every dimension
	 */
	public ArrayList<Double> getMeanAL()
	{
		ArrayList<Double> means	= new ArrayList<Double>(statistics.size());
		for (ContStats dimension : statistics)
			means.add(dimension.getMean());
		return means;
	}

	@Override
	public double[] getMean()
	{
		ArrayList<Double> meanAL	= getMeanAL();
		double[] mean				= new double[meanAL.size()];
		for (int m = 0; m < meanAL.size(); m++)
			mean[m]					= meanAL.get(m);
		return mean;
	}

	@Override
	public double[] getVariance()
	{
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public double[][] getCovariance()
	{
		// TODO Auto-generated method stub
		return null;
	}
	
	/**
	 * Determines the kernels of the distribution by using a lightweight Gaussian graphical model 
	 * ({@link GGMLite}) with default parameters and using Silverman's rule of thumb for scaling
	 * the sample covariance values
	 */
	public void computeGaussianBW()
	{
		computeGaussianBW(getSilvermansScaling(), GGMLite.DEF_CLIQUE_RATIO, 
				GGMLite.DEF_MAX_CLIQUE_SIZE, GGMLite.DEF_SHARE_PERCENT, GGMLite.DEF_FREE_R2_THRESH,
				GGMLite.DEF_FREE_VAR_THRESH, GGMLite.DEF_DETERM_R2_THRESH,
				GGMLite.DEF_DETERM_VAR_THRESH, GGMLite.DEF_RANDOM_BUILD, GGMLite.DEF_MIN_VARIANCE);
	}
	
	/**
	 * Determines the kernels of the distribution by using a lightweight Gaussian graphical model 
	 * ({@link GGMLite}) with default parameters
	 * @param scaling	The scalar factor to multiply the sample covariance matrix to obtain the
	 * 					bandwidth matrix
	 */
	public void computeGaussianBW(double scaling)
	{
		computeGaussianBW(scaling, GGMLite.DEF_CLIQUE_RATIO, GGMLite.DEF_MAX_CLIQUE_SIZE,
				GGMLite.DEF_SHARE_PERCENT, GGMLite.DEF_FREE_R2_THRESH, GGMLite.DEF_FREE_VAR_THRESH,
				GGMLite.DEF_DETERM_R2_THRESH, GGMLite.DEF_DETERM_VAR_THRESH,
				GGMLite.DEF_RANDOM_BUILD, GGMLite.DEF_MIN_VARIANCE);
	}
	
	/**
	 * Determines the kernels of the distribution by using a lightweight Gaussian graphical model 
	 * ({@link GGMLite}) with the provided parameters and using Silverman's rule of thumb for 
	 * scaling the sample covariance values
	 * @param scaling	The scalar factor to multiply the sample covariance matrix to obtain the
	 * 					bandwidth matrix
	 * @param creator	An object containing the parameters for the kernel
	 */
	public void computeGaussianBW(double scaling, GGMLiteCreator creator)
	{
		computeGaussianBW(scaling, creator.getCliqueRatio(), creator.getMaxCliqueSize(), 
				creator.getSharePercent(), creator.getFreeR2Thresh(), creator.getFreeVarThresh(),
				creator.getDetermR2Thresh(), creator.getDetermVarThresh(), 
				creator.getRandomBuild(), creator.getMinVariance());
	}
	
	/**
	 * Determines the kernels of the distribution by using a lightweight Gaussian graphical model 
	 * ({@link GGMLite}) with the provided parameters and using Silverman's rule of thumb for 
	 * scaling the sample covariance values
	 * @param cliqueRatio		The ratio of <i>n</i> to limit the size of the cliques (since the 
	 * 							rank of covariance matrices is limited by <i>n</i>). Should be 
	 * 							larger than zero.
	 * @param maxCliqueSize		A hard limit on the size of the cliques which controls the size of
	 * 							the covariance matrices used to relate co-dependent variables
	 * @param sharePercent		The percentage of variables to be shared by cliques (and thus to 
	 * 							maintain their interconnection)
	 * @param freeR2Thresh		The maximum value for the coefficient of determination of the
	 * 							regression based on other variables to consider a variable 
	 * 							free/independent
	 * @param freeVarThresh		The maximum value for the percentage of variance explained by the
	 * 							regression to consider a variable free/independent
	 * @param determR2Thresh	The minimum value for the coefficient of determination of the
	 * 							regression based on other variables to consider a variable fully
	 * 							determined/dependent
	 * @param determVarThresh	The minimum value for the percentage of variance explained by the
	 * 							regression to consider a variable fully	determined/dependent
	 * @param randomBuild		True if the clique array should be built by adding variables in a
	 * 							random order
	 * @param minVariance		{@link GGMLite#minVariance}
	 */
	public void computeGaussianBW(double scaling, double cliqueRatio, int maxCliqueSize,
			double sharePercent, double freeR2Thresh, double freeVarThresh, double determR2Thresh,
			double determVarThresh, boolean randomBuild, double minVariance)
	{
		this.scaling	= scaling;
		kernel			= new GGMLite(samples, cliqueRatio, maxCliqueSize, sharePercent, 
										freeR2Thresh, freeVarThresh, determR2Thresh,
										determVarThresh, randomBuild, minVariance);
	}
	
	/**
	 * @return The scaling factor for the bandwidth matrix according to Silverman's rule for 
	 * assumed underlying Gaussian distributions as described in the reference below. The criteria 
	 * is termed "normal distribution approximation" or "Silverman's rule of thumb".
	 * <br><br>
	 * Silverman, B.W., 1998, <i>Density Estimation for Statistics and Data Analysis</i>, London: 
	 * Chapman & Hall/CRC, p.48, ISBN 0-412-24620-1.
	 * <p>Time complexity: <i>O(1)</i>
	 * <br>Space complexity: <i>O(1)</i>
	 */
	public double getSilvermansScaling()
	{
		int sampleCount	= samples.size();
		int dimensions	= statistics.size();
		double temp1	= Math.pow(sampleCount, -1.0/(dimensions + 4));
		double temp2	= Math.pow(4.0/(dimensions + 2), 1.0/(dimensions + 4));
		return temp1*temp1*temp2*temp2;
	}

	@Override
	public double getpdf(ArrayList<Double> x)
	{
		return Math.exp(getLogpdf(x));
	}

	@Override
	public double getpdf(double[] x)
	{
		return Math.exp(getLogpdf(x));
	}

	@Override
	public double getLogpdf(ArrayList<Double> x)
	{
		return getLogpdf(Utilities.toArray(x));
	}

	@Override
	public double getLogpdf(double[] x)
	{
		int k						= statistics.size();
		if (samples.size() == 0)			
			throw new RuntimeException(ERR_NO_SAMPLES);
		if (x.length < k)
			throw new IllegalArgumentException(String.format(ERR_PARTICLE_SIZE, x.length, k));
		
		double logSum				= 0.0;
		double minStDev				= Math.sqrt(kernel.getMinVariance());
		
		// Constants
		for (Integer c : kernel.getConstants())
		{
			double diff				= Math.abs(x[c] - statistics.get(c).getMean());
			if (diff > minStDev)
				return Double.NEGATIVE_INFINITY;
		}
		
		// Free
		for (Integer f : kernel.getFree())
		{
			ContStats stat			= statistics.get(f);
			Normal dist				= new Normal(stat.getMean(), stat.getStDev());
			double logpdf			= Math.log(dist.getpdf(x[f]));
			if (Double.isFinite(logpdf))
				logSum				+= logpdf;
			else
				return Double.NEGATIVE_INFINITY;
		}
		
		// Determined
		Hashtable<Integer, ArrayList<PointID>> reg = kernel.getRegCoeff();
		for (Integer d : reg.keySet())
		{
			double combination		= 0.0;
			double stDev			= Double.NEGATIVE_INFINITY;
			for (PointID coeff : reg.get(d))
			{
				int index			= coeff.x;
				if (index == -2)
					stDev = coeff.y;
				else
					combination		+= index == -1 ? coeff.y : coeff.y*x[index];
			}
			Normal dist				= new Normal(combination, Math.max(stDev, minStDev));
			double logpdf			= Math.log(dist.getpdf(x[d]));
			if (Double.isFinite(logpdf))
				logSum				+= logpdf;
			else
				return Double.NEGATIVE_INFINITY;
		}
		
		// Cliques
		double sum					= 0.0;
		double weightSum			= 0.0;
		for (ContMultiSample sample : samples)
		{
			ArrayList<Double> vals	= sample.getValues();
			double[] transformed	= new double[k];
			for (int d = 0; d < k; d++)
			{
				double value		= vals.get(d);
				double deviation	= x[d] - value;
				transformed[d]		= statistics.get(d).getMean() + deviation/Math.sqrt(scaling);
			}
			double logpdfSamp		= kernel.getLogpdfCliques(transformed);
			double contrib			= Math.exp(logpdfSamp);
			double weight			= weighted ? sample.getWeight() : 1.0;
			sum						+= contrib*weight;
			weightSum				+= weight;
			if (contrib == 0.0)
				return Double.NEGATIVE_INFINITY;
		}
		logSum						+= Math.log(sum/weightSum);
		
		return logSum;
	}
	
	@Override
	public double getMeanMahalanobisDistance(double[] x)
	{
		int k						= statistics.size();
		if (samples.size() == 0)			
			throw new RuntimeException(ERR_NO_SAMPLES);
		if (x.length < k)
			throw new IllegalArgumentException(String.format(ERR_PARTICLE_SIZE, x.length, k));
		ContStats distanceSeries	= new ContStats(weighted);
		
		for (ContMultiSample sample : samples)
		{
			ArrayList<Double> vals	= sample.getValues();
			double[] transformed	= new double[k];
			for (int d = 0; d < k; d++)
			{
				double value		= vals.get(d);
				double deviation	= x[d] - value;
				transformed[d]		= statistics.get(d).getMean() + deviation/Math.sqrt(scaling);
			}
			double distance			= kernel.getMahalanobisDistance(transformed);
			double weight			= weighted ? sample.getWeight() : 1.0;
			distanceSeries.addValue(distance, weight);
		}
		return distanceSeries.getMean();
	}
	
	@Override
	public double[] getMahalanobisDistanceToSamples(double[] x)
	{
		int k						= statistics.size();
		int sampleCount				= samples.size();
		if (sampleCount == 0)			
			throw new RuntimeException(ERR_NO_SAMPLES);
		if (x.length < k)
			throw new IllegalArgumentException(String.format(ERR_PARTICLE_SIZE, x.length, k));
		double distances[]			= new double[sampleCount];
		
		for (int s = 0; s < sampleCount; s++)
		{
			ContMultiSample sample	= samples.get(s);
			ArrayList<Double> vals	= sample.getValues();
			double[] transformed	= new double[k];
			for (int d = 0; d < k; d++)
			{
				double value		= vals.get(d);
				double deviation	= x[d] - value;
				transformed[d]		= statistics.get(d).getMean() + deviation/Math.sqrt(scaling);
			}
			distances[s]			= kernel.getMahalanobisDistance(transformed);
		}
		return distances;
	}

	@Override
	public double getMeanMahalanobisForce(double[] x)
	{
		int k						= statistics.size();
		if (samples.size() == 0)			
			throw new RuntimeException(ERR_NO_SAMPLES);
		if (x.length < k)
			throw new IllegalArgumentException(String.format(ERR_PARTICLE_SIZE, x.length, k));
		ContStats forceSeries		= new ContStats(weighted);
		
		for (ContMultiSample sample : samples)
		{
			ArrayList<Double> vals	= sample.getValues();
			double[] transformed	= new double[k];
			for (int d = 0; d < k; d++)
			{
				double value		= vals.get(d);
				double deviation	= x[d] - value;
				transformed[d]		= statistics.get(d).getMean() + deviation/Math.sqrt(scaling);
			}
			double distance			= kernel.getMahalanobisDistance(transformed);
			double force			= 1/(distance*distance);  
			double weight			= weighted ? sample.getWeight() : 1.0;
			forceSeries.addValue(force, weight);
		}
		return forceSeries.getMean();
	}

	@Override
	public double getMahalanobisForce(double[] x)
	{
		int k						= statistics.size();
		if (samples.size() == 0)			
			throw new RuntimeException(ERR_NO_SAMPLES);
		if (x.length < k)
			throw new IllegalArgumentException(String.format(ERR_PARTICLE_SIZE, x.length, k));
		double sqrdDistance			= 0.0;
		
		// Free
		for (Integer f : kernel.getFree())
		{
			ContStats stat			= statistics.get(f);
			double diff				= stat.getMean() - x[f];
			if (diff != 0.0)
			{
				double var			= Math.max(stat.getVar(), kernel.getMinVariance());
				sqrdDistance		+= diff*diff/var;
			}
		}
		
		// Determined
		Hashtable<Integer, ArrayList<PointID>> reg = kernel.getRegCoeff();
		for (Integer d : reg.keySet())
		{
			double combination		= 0.0;
			double stDev			= Double.NEGATIVE_INFINITY;
			for (PointID coeff : reg.get(d))
			{
				int index			= coeff.x;
				if (index == -2)
					stDev			= coeff.y;
				else
					combination		+= index == -1 ? coeff.y : coeff.y*x[index];
			}
			double diff				= combination - x[d];
			if (diff != 0.0)
			{
				stDev				= Math.max(stDev, Math.sqrt(kernel.getMinVariance()));
				sqrdDistance		+= diff*diff/(stDev*stDev);
			}
		}
		
		// Cliques
		double sum					= 0.0;
		double weightSum			= 0.0;
		for (ContMultiSample sample : samples)
		{
			ArrayList<Double> vals	= sample.getValues();
			double[] transformed	= new double[k];
			for (int d = 0; d < k; d++)
			{
				double value		= vals.get(d);
				double deviation	= x[d] - value;
				transformed[d]		= statistics.get(d).getMean() + deviation/Math.sqrt(scaling);
			}
			double sqrdDist			= kernel.getSqrdMahalanobisDistCliques(transformed);
			double weight			= weighted ? sample.getWeight() : 1.0;
			sum						+= sqrdDist/weight;
			weightSum				+= weight;
		}
		sqrdDistance				+= sum*weightSum;
		
		return 1/sqrdDistance;
	}

	@Override
	public double getMeanIndeppdf(double[] x)
	{
		int k						= statistics.size();
		if (samples.size() == 0)			
			throw new RuntimeException(ERR_NO_SAMPLES);
		if (x.length < k)
			throw new IllegalArgumentException(String.format(ERR_PARTICLE_SIZE, x.length, k));
		ContStats pdfs				= new ContStats(false);
		
		for (int d = 0; d < k; d++)
		{
			ContStats series		= statistics.get(d);
			double mean				= series.getMean();
			double variance			= (1.0 + scaling)*series.getVar();
			if (variance > 0.0)
			{
				double pdf			= Normal.computepdf(mean, Math.sqrt(variance), x[d]);
				pdfs.addValue(pdf);
			}
		}
		return pdfs.getMean();
	}

	@Override
	public ContProbDist marginalize(int dimension)
	{
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public MultiContProbDist marginalize(ArrayList<Integer> toActivate)
	{
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public MultiContProbDist conditional(ArrayList<Double> x)
	{
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public double[] sample()
	{
		if (samples.size() == 0)			
			throw new RuntimeException(ERR_NO_SAMPLES);
		
		int k						= statistics.size();
		double[] randomSample		= new double[k];
		
		// Identify covariates
		HashSet<Integer> covariates	= new HashSet<>(k);
		for (int d = 0; d < k; d++)
			covariates.add(d);
		Hashtable<Integer, ArrayList<PointID>> reg = kernel.getRegCoeff();
		for (Integer dep : reg.keySet())
			covariates.remove(dep);
		
		// Constants
		for (Integer index : kernel.getConstants())
		{
			randomSample[index]		= statistics.get(index).getMean();
			covariates.remove(index);
		}
		
		// Free
		for (Integer index : kernel.getFree())
		{
			ContStats stats			= statistics.get(index);
			randomSample[index]		= Normal.sample(stats.getMean(), stats.getStDev());
			covariates.remove(index);
		}
		
		// Select sample
		double weightSum			= statistics.get(0).getWeightSum();
		double root					= Math.random()*weightSum;
		double sum 					= 0;
		int index					= -1;
		ContMultiSample sample		= null;
		while (root >= sum || sample == null)
		{
			index++;
			sample					= (ContMultiSample) samples.get(index);
			sum						+= sample.getWeight();
		}
		double[] mean				= Utilities.toArray(sample.getValues());
		
		// Sample kernel
		double[] kernelSamp			= kernel.sample();
		for (Integer covariate : covariates)
		{
			double deviation		= kernelSamp[covariate] - statistics.get(covariate).getMean();
			randomSample[covariate]	= mean[covariate] + deviation*Math.sqrt(scaling);
		}
		
		// Dependent
		for (Integer dep : reg.keySet())
		{
			double stDev			= 0.0;
			double predicted		= 0.0;
			for (PointID coeff : reg.get(dep))
			{
				int index2			= coeff.x;
				if (index2 == -2)
					stDev			= coeff.y;
				else 
					predicted		+= coeff.y*(index2 == -1 ? 1.0 : randomSample[index2]);
			}
			randomSample[dep]		= stDev > 0.0 ? Normal.sample(predicted, stDev) : predicted;
		}
		
		return randomSample;
	}
	
	@Override
	public double[][] sampleMultiple(int count)
	{
		if (samples.size() == 0)			
			throw new RuntimeException(ERR_NO_SAMPLES);
		
		int k						= statistics.size();
		double[][] randomSamples	= new double[count][k];
		for (int s = 0; s < count; s++)
			randomSamples[s]		= sample();
		return randomSamples;
	}
	
	/**
	 * Generates a defined number of random samples from this distribution
	 * @param count The number of random samples to generate
	 * @return A list of random samples from this distribution
	 */
	public ArrayList<Sample> sampleMultipleOb(int count)
	{
		double[][] samples		= sampleMultiple(count);
		ArrayList<Sample> array	= new ArrayList<>(count);
		for (int s = 0; s < count; s++)
			array.add(new Sample(1.0, Utilities.toArrayList(samples[s])));
		return array;
	}
	
}
