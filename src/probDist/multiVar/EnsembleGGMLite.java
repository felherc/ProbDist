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
 * This class represents multivariate normal (Gaussian) probability distributions which are
 * parameterized by a vector of mean values and a covariance matrix. The distribution is 
 * constructed from a series of samples. The covariance matrix is represented by a series of
 * interconnected cliques of lower rank to improve storage and inference efficiency. Due to their
 * sparse representation, they fall under the category of Gaussian Graphical Models (GGMs) and, due
 * to their resource frugality, they earn the modifier "lite."
 * @author Felipe Hernández
 */
public class EnsembleGGMLite extends NonParametric
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
	 * A lightweight Gaussian graphical model that represents the distribution
	 */
	private GGMLite distribution;
	
	// --------------------------------------------------------------------------------------------
	// Constructors
	// --------------------------------------------------------------------------------------------
	
	public EnsembleGGMLite()
	{
		type				= MultiContProbDist.KERNEL_GGMLITE;
		weighted			= false;
		samples				= new ArrayList<>();
		statistics			= new ArrayList<>();
		distribution				= null;
	}
	
	/**	
	 * @param weighted {@link #weighted}
	 */
	public EnsembleGGMLite(boolean weighted)
	{
		type				= MultiContProbDist.KERNEL_GGMLITE;
		this.weighted		= weighted;
		samples				= new ArrayList<>();
		statistics			= new ArrayList<>();
		distribution				= null;
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
		distribution = null;
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
	 * @return {@link #distribution}
	 */
	public GGMLite getKernel()
	{
		return distribution;
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
			double value			= values.get(i);
			if (!Double.isNaN(value))
				statistics.get(i).addValue(value, weight);
		}
		distribution				= null;
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
	 * Determines the dependencies between variables by using a lightweight Gaussian graphical
	 * model ({@link GGMLite}) with default parameters
	 */
	public void computeDependencies()
	{
		computeDependencies(GGMLite.DEF_CLIQUE_RATIO, GGMLite.DEF_MAX_CLIQUE_SIZE,
				GGMLite.DEF_SHARE_PERCENT, GGMLite.DEF_FREE_R2_THRESH, GGMLite.DEF_FREE_VAR_THRESH,
				GGMLite.DEF_DETERM_R2_THRESH, GGMLite.DEF_DETERM_VAR_THRESH,
				GGMLite.DEF_RANDOM_BUILD, GGMLite.DEF_MIN_VARIANCE);
	}
	
	/**
	 * Determines the dependencies between variables by using a lightweight Gaussian graphical
	 * model ({@link GGMLite}) with the provided parameters
	 * @param creator	An object containing the parameters for the kernel
	 */
	public void computeDependencies(GGMLiteCreator creator)
	{
		computeDependencies(creator.getCliqueRatio(), creator.getMaxCliqueSize(), 
				creator.getSharePercent(), creator.getFreeR2Thresh(), creator.getFreeVarThresh(),
				creator.getDetermR2Thresh(), creator.getDetermVarThresh(), 
				creator.getRandomBuild(), creator.getMinVariance());
	}
	
	/**
	 * Determines the dependencies between variables by using a lightweight Gaussian graphical
	 * model ({@link GGMLite}) with the provided parameters
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
	public void computeDependencies(double cliqueRatio, int maxCliqueSize, double sharePercent,
			double freeR2Thresh, double freeVarThresh, double determR2Thresh,
			double determVarThresh, boolean randomBuild, double minVariance)
	{
		distribution = new GGMLite(samples, cliqueRatio, maxCliqueSize, sharePercent, freeR2Thresh,
						freeVarThresh, determR2Thresh, determVarThresh, randomBuild, minVariance);
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
		double minStDev				= Math.sqrt(distribution.getMinVariance());
		
		// Constants
		for (Integer c : distribution.getConstants())
		{
			double diff				= Math.abs(x[c] - statistics.get(c).getMean());
			if (diff > minStDev)
				return Double.NEGATIVE_INFINITY;
		}
		
		// Free
		for (Integer f : distribution.getFree())
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
		Hashtable<Integer, ArrayList<PointID>> reg = distribution.getRegCoeff();
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
		logSum						+= distribution.getLogpdfCliques(x);
		
		return logSum;
	}
	
	@Override
	public double getMeanMahalanobisDistance(double[] x)
	{
		int k = statistics.size();
		if (samples.size() == 0)			
			throw new RuntimeException(ERR_NO_SAMPLES);
		if (x.length < k)
			throw new IllegalArgumentException(String.format(ERR_PARTICLE_SIZE, x.length, k));
			
		return distribution.getMahalanobisDistance(x);
	}
	
	@Override
	public double[] getMahalanobisDistanceToSamples(double[] x)
	{
		return null;
	}

	@Override
	public double getMeanMahalanobisForce(double[] x)
	{
		int k			= statistics.size();
		if (samples.size() == 0)			
			throw new RuntimeException(ERR_NO_SAMPLES);
		if (x.length < k)
			throw new IllegalArgumentException(String.format(ERR_PARTICLE_SIZE, x.length, k));
		
		double distance	= distribution.getMahalanobisDistance(x);
		return 1/(distance*distance);
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
		for (Integer f : distribution.getFree())
		{
			ContStats stat			= statistics.get(f);
			double diff				= stat.getMean() - x[f];
			if (diff != 0.0)
			{
				double var			= Math.max(stat.getVar(), distribution.getMinVariance());
				sqrdDistance		+= diff*diff/var;
			}
		}
		
		// Determined
		Hashtable<Integer, ArrayList<PointID>> reg = distribution.getRegCoeff();
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
				stDev				= Math.max(stDev, Math.sqrt(distribution.getMinVariance()));
				sqrdDistance		+= diff*diff/(stDev*stDev);
			}
		}
		
		// Cliques
		sqrdDistance				+= distribution.getSqrdMahalanobisDistCliques(x);
		
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
			double variance			= series.getVar();
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
		Hashtable<Integer, ArrayList<PointID>> reg = distribution.getRegCoeff();
		for (Integer dep : reg.keySet())
			covariates.remove(dep);
		
		// Constants
		for (Integer index : distribution.getConstants())
		{
			randomSample[index]		= statistics.get(index).getMean();
			covariates.remove(index);
		}
		
		// Free
		for (Integer index : distribution.getFree())
		{
			ContStats stats			= statistics.get(index);
			randomSample[index]		= Normal.sample(stats.getMean(), stats.getStDev());
			covariates.remove(index);
		}
		
		// Sample kernel
		double[] kernelSamp			= distribution.sample();
		for (Integer covariate : covariates)
			randomSample[covariate]	= kernelSamp[covariate];
		
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
