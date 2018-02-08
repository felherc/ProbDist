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
import java.util.Hashtable;
import java.util.concurrent.ArrayBlockingQueue;

import probDist.Normal;
import probDist.multiVar.tools.Clique;
import probDist.multiVar.tools.ContMultiSample;
import probDist.multiVar.tools.StatsCalculator;
import utilities.MatUtil;
import utilities.geom.Point2I;
import utilities.geom.PointID;
import utilities.stat.ContPairedSeries;
import utilities.stat.ContStats;

/**
 * This class represents multivariate normal (Gaussian) probability distributions which are
 * parameterized by a vector of mean values and a covariance matrix. The covariance matrix is
 * represented by a series of interconnected cliques of lower rank to improve storage and inference
 * efficiency. Due to their sparse representation, they fall under the category of Gaussian
 * Graphical Models (GGMs) and, due to their resource frugality, they earn the modifier "lite."
 * @author Felipe Hernández
 */
public class GGMLite extends MultiContProbDist implements StatsCalculator
{
	
	// --------------------------------------------------------------------------------------------
	// Constants
	// --------------------------------------------------------------------------------------------
	
	/**
	 * Distribution type identifier
	 */
	public final static String ID = "Lightweight Gaussian graphical model";
	
	/**
	 * Default value for the ratio of <i>n</i> to limit the size of the cliques
	 */
	public final static double DEF_CLIQUE_RATIO = 1.0;
	
	/**
	 * Default value for the hard limit on the size of the cliques which controls the size of the
	 * covariance matrices used to relate co-dependent variables
	 */
	public final static int DEF_MAX_CLIQUE_SIZE = Integer.MAX_VALUE;

	/**
	 * Default value for the percentage of variables to be shared by cliques (and thus maintain 
	 * their interconnection)
	 */
	public final static double DEF_SHARE_PERCENT = 0.25;
	
	/**
	 * Default value for the maximum value for the coefficient of determination of the regression
	 * based on other variables to consider a variable free/independent
	 */
	public final static double DEF_FREE_R2_THRESH = 0.25;
	
	/**
	 * Default value for the percentage of variance explained by the regression to consider a
	 * variable fully determined/dependent
	 */
	public final static double DEF_FREE_VAR_THRESH = 0.25;
	
	/**
	 * Default value for the coefficient of determination of the regression based on other
	 * variables to consider a variable fully determined/dependent
	 */
	public final static double DEF_DETERM_R2_THRESH = 0.75;
	
	/**
	 * Default value for the percentage of variance explained by the regression to consider a 
	 * variable fully determined/dependent
	 */
	public final static double DEF_DETERM_VAR_THRESH = 0.75;

	/**
	 * Default value for whether if the clique array should be built by adding variables in a
	 * random order
	 */
	public final static boolean DEF_RANDOM_BUILD = true;
	
	/**
	 * Default value for {@link #minVariance}
	 */
	public final static double DEF_MIN_VARIANCE = 1E-10;
	
	/**
	 * Variable type for the {@link #variableIndex}: variable is constant
	 */
	public final static int VAR_TYPE_CONSTANT = -3;
	
	/**
	 * Variable type for the {@link #variableIndex}: variable is free/independent
	 */
	public final static int VAR_TYPE_FREE = -2;
	
	/**
	 * Variable type for the {@link #variableIndex}: variable is determined/fully-dependent
	 */
	public final static int VAR_TYPE_DETERMINED = -1;
	
	/**
	 * Error message: mean and C size mismatch
	 */
	public final static String ERR_MEAN_C_SIZE_MISMATCH = "The mean and the covariance matrix "
															+ "have mismatching sizes";

	/**
	 * Error message: mean and variance size mismatch
	 */
	public final static String ERR_MEAN_VARIANCE_SIZE_MISMATCH = "The mean and variance vectors "
																	+ "do not have the same size";

	/**
	 * Error message: C not symmetrical
	 */
	public final static String ERR_C_NOT_SYMMETRICAL = "The covariance matrix is not symmetrical";

	/**
	 * Error message: C not square
	 */
	public final static String ERR_C_NOT_SQUARE = "The covariance matrix is not square";
	
	/**
	 * Error message: Not enough samples
	 */
	public final static String ERR_NOT_ENOUGH_SAMPLES = "There must be at least two samples "
															+ "(there are %1$d)";
	
	/**
	 * Error message: samples are empty
	 */
	public final static String ERR_NO_VARIABLES = "The samples contain no variables";

	/**
	 * Error message: clique ratio too small
	 */
	public final static String ERR_CLIQUE_RATIO_TOO_SMALL = "The clique ratio should be larger "
																+ "than zero (it is %1$f)";

	/**
	 * Error message: maximum clique size too small
	 */
	public final static String ERR_MAX_CLIQUE_SIZE_TOO_SMALL = "The maximum clique size should "
															+ "be larger than 2 (it is %1$d)";

	/**
	 * Error message: invalid share percentage
	 */
	public final static String ERR_INVALID_SHARE_PERCENT = "The shared percentage should be at "
											+ "least zero and smaller than one (it is %1$f)";
	
	/**
	 * Error message: invalid sample size
	 */
	public final static String ERR_SAMPLE_SIZE = "The sample size is %1$f but it should be %2$f";
	
	/**
	 * Error message: invalid vector size
	 */
	public final static String ERR_VECTOR_SIZE = "The provided vector size is %1$f but it should"
													+ " be %2$f";
	
	// --------------------------------------------------------------------------------------------
	// Attributes
	// --------------------------------------------------------------------------------------------
	
	/**
	 * The vector with the mean value of each dimension/variable <i>k</i>
	 */
	private double[] mean;
	
	/**
	 * The vector with the marginal variance of each dimension/variable <i>k</i>, that is the 
	 * coefficients in the diagonal of the covariance matrix. Represents a distribution of 
	 * independent variables (covariance values assumed to be zero).
	 */
	private double[] variance;
	
	/**
	 * The indices of the dimensions/variables that have zero variance and, therefore, cannot be 
	 * used for likelihood and sampling computations using inverse covariance matrix methods.
	 */
	private ArrayList<Integer> constant;
	
	/**
	 * The indices of the dimensions/variables that are independent from all others
	 */
	private ArrayList<Integer> free;
	
	/**
	 * Indexes each of the dimensions/variables in the distribution:<ul>
	 * <li>Constant: Only value = {@link #VAR_TYPE_CONSTANT}
	 * <li>Free/independent: Only value = {@link #VAR_TYPE_FREE}
	 * <li>Determined/fully-dependent: Only value = {@link #VAR_TYPE_DETERMINED}
	 * <li>Co-dependent: the index of each clique it belongs to</ul>
	 */
	private Hashtable<Integer, ArrayList<Integer>> variableIndex;
	
	/**
	 * The set of cliques that represent the correlated dimensions/variables of the distribution
	 */
	private ArrayList<Clique> cliques;
	
	/**
	 * The indices of the dimensions that have a linear dependency on other variables in the 
	 * distribution and, therefore, cannot be used for likelihood and sampling computations using
	 * inverse covariance matrix methods
	 */
	private ArrayList<Integer> linearlyDependent;
	
	/**
	 * Stores the regression coefficients to predict the variables of the distribution from the
	 * others assuming a linear combination. The key of the hash table is the index of the
	 * dependent variable, and the list contains tuples with the index of the explanatory variables
	 * and the corresponding coefficient. The -1 index in the tuples represents the constant 
	 * coefficient. The -2 index represents the free/unexplained variance of the variable being
	 * predicted.
	 */
	private Hashtable<Integer, ArrayList<PointID>> regressionCoeff;
	
	/**
	 * Minimum allowed value for the covariance matrix before it is considered zero for likelihood
	 * and sampling computation purposes. Also used as the minimum value to assume that a
	 * determinant is practically zero (the matrix is singular), and that a mean value for a
	 * dimension of the distribution is practically zero.
	 */
	private double minVariance;
	
	/**
	 * Stores the root samples while the distribution is being constructed
	 */
	private ArrayList<ContMultiSample> samples;
	
	/**
	 * Stores covariance values while the distribution is being constructed, indexed by the
	 * variables' indices
	 */
	private Hashtable<Point2I, Double> covIndex;
	
	/**
	 * Indicates the adding order of values into {@link #covIndex}
	 */
	private ArrayBlockingQueue<Point2I> covQueue;
	
	/**
	 * The target maximum size of {@link #covIndex}
	 */
	private int covIndexSize;
	
	// --------------------------------------------------------------------------------------------
	// Constructors
	// --------------------------------------------------------------------------------------------
	
	/**
	 * Creates a distribution where all variables are assumed to be independent
	 * @param mean {@link #mean}
	 * @param variance {@link #variance}
	 */
	public GGMLite(double[] mean, double[] variance)
	{
		type				= MultiContProbDist.GGMLITE;
		
		if (mean.length != variance.length)
			throw new IllegalArgumentException(ERR_MEAN_VARIANCE_SIZE_MISMATCH + ": mean.length = " 
										+ mean.length + "; variance.length = " + variance.length);
		
		this.mean			= mean;
		this.variance		= variance;
		
		constant			= new ArrayList<>(mean.length);
		free				= new ArrayList<>(mean.length);
		for (int v = 0; v < mean.length; v++)
			if (variance[v] < minVariance)
				constant.add(v);
			else
				free.add(v);
		
		variableIndex		= new Hashtable<>();
		cliques				= new ArrayList<>();
		linearlyDependent	= new ArrayList<>();
		regressionCoeff		= new Hashtable<>();

		minVariance			= DEF_MIN_VARIANCE;
	}
	
	/**
	 * Creates a new distribution based on set of <i>n</i> samples using the default parameters
	 * @param samples	The list of samples
	 */
	public GGMLite(ArrayList<ContMultiSample> samples)
	{
		initialize(samples, DEF_CLIQUE_RATIO, DEF_MAX_CLIQUE_SIZE, DEF_SHARE_PERCENT, 
						DEF_FREE_R2_THRESH, DEF_FREE_VAR_THRESH, DEF_DETERM_R2_THRESH, 
						DEF_DETERM_VAR_THRESH, DEF_RANDOM_BUILD, DEF_MIN_VARIANCE);
	}
	
	/**
	 * Creates a new distribution based on set of <i>n</i> samples
	 * @param samples			The samples of the underlying distribution to model. Samples can
	 * 							have missing values, which should be set to 
	 * 							{@link java.lang.Double#NaN}.
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
	 * @param freeVarThres		The maximum value for the percentage of variance explained by the
	 * 							regression to consider a variable free/independent
	 * @param determR2Thresh	The minimum value for the coefficient of determination of the
	 * 							regression based on other variables to consider a variable fully
	 * 							determined/dependent
	 * @param determVarThresh	The minimum value for the percentage of variance explained by the
	 * 							regression to consider a variable fully	determined/dependent
	 * @param randomBuild		True if the clique array should be built by adding variables in a
	 * 							random order
	 * @param minVariance		{@link #minVariance}
	 */
	public GGMLite(ArrayList<ContMultiSample> samples, double cliqueRatio, 
			int maxCliqueSize, double sharePercent, double freeR2Thresh, double freeVarThresh,
			double determR2Thresh, double determVarThresh, boolean randomBuild, double minVariance)
	{
		initialize(samples, cliqueRatio, maxCliqueSize, sharePercent, freeR2Thresh, freeVarThresh, 
						determR2Thresh, determVarThresh, randomBuild, minVariance);
	}
	
	/**
	 * @param mean {@link #mean}
	 * @param C {@link #C}
	 */
	public GGMLite(double[] mean, double[][] C)
	{
		type = MultiContProbDist.GGMLITE;
		
		if (!MatUtil.isSquare(C))
			throw new IllegalArgumentException(ERR_C_NOT_SQUARE);
		
		MatUtil.enforceSymmetry(C);
		
		if (mean.length != C.length)
			throw new IllegalArgumentException(ERR_MEAN_C_SIZE_MISMATCH + ": mean.length = " 
												+ mean.length + "; C.length = " + C.length);
		
		minVariance			= DEF_MIN_VARIANCE;
	}
	
	/**
	 * Determines which variables are constant, free, co-dependent, and determined based on set of
	 * <i>n</i> samples
	 * @param samples			The samples of the underlying distribution to model. Samples can
	 * 							have missing values, which should be set to 
	 * 							{@link java.lang.Double#NaN}.
	 * @param cliqueRatio		The ratio of <i>n</i> to limit the size of the cliques (since the 
	 * 							rank of covariance matrices is limited by <i>n</i>). Should be 
	 * 							larger than zero.
	 * @param maxCliqueSize		A hard limit on the size of the cliques which controls the size of
	 * 							the covariance matrices used to relate co-dependent variables
	 * @param sharePercent		The percentage of variables to be shared by cliques (and thus 
	 * 							maintain their interconnection)
	 * @param freeR2Thresh		The maximum value for the coefficient of determination of the
	 * 							regression based on other variables to consider a variable 
	 * 							free/independent
	 * @param freeVarThres		The maximum value for the percentage of variance explained by the
	 * 							regression to consider a variable free/independent
	 * @param determR2Thresh	The minimum value for the coefficient of determination of the
	 * 							regression based on other variables to consider a variable fully
	 * 							determined/dependent
	 * @param determVarThresh	The minimum value for the percentage of variance explained by the
	 * 							regression to consider a variable fully	determined/dependent
	 * @param randomBuild		True if the clique array should be built by adding variables in a
	 * 							random order
	 * @param minVariance		{@link #minVariance}
	 */
	private void initialize(ArrayList<ContMultiSample> samples, double cliqueRatio, 
			int maxCliqueSize, double sharePercent, double freeR2Thresh, double freeVarThresh,
			double determR2Thresh, double determVarThresh, boolean randomBuild, double minVariance)
	{
		long start							= System.currentTimeMillis();
		
		type								= MultiContProbDist.GGMLITE;
		this.minVariance					= minVariance;
		
		// Verify parameters
		int n								= samples.size();
		int dimensions						= samples.get(0).getValues().size();
		if (n < 2)
			throw new IllegalArgumentException(
							String.format(ERR_NOT_ENOUGH_SAMPLES,			n				));
		if (dimensions == 0)
			throw new IllegalArgumentException(ERR_NO_VARIABLES);
		if (cliqueRatio <= 0.0)
			throw new IllegalArgumentException(
							String.format(ERR_CLIQUE_RATIO_TOO_SMALL,		cliqueRatio		));
		if (maxCliqueSize < 2)
			throw new IllegalArgumentException(
							String.format(ERR_MAX_CLIQUE_SIZE_TOO_SMALL,	maxCliqueSize	));
		if (sharePercent < 0.0 || sharePercent >= 1.0)
			throw new IllegalArgumentException(
							String.format(ERR_INVALID_SHARE_PERCENT,		sharePercent	));
		
		// Analyze samples
		ArrayList<ContStats> statistics	= new ArrayList<>(dimensions);
		for (int d = 0; d < dimensions; d++)
			statistics.add(new ContStats(true));
		for (ContMultiSample sample : samples)
		{
			double weight					= sample.getWeight();
			ArrayList<Double> values		= sample.getValues();
			for (int v = 0; v < values.size(); v++)
			{
				double value				= values.get(v);
				if (!Double.isNaN(value))
					statistics.get(v).addValue(value, weight);
			}
		}
		
		// Initialize arrays
		mean								= new double[dimensions];
		variance							= new double[dimensions];
		constant							= new ArrayList<>(dimensions);
		free								= new ArrayList<>(dimensions);
		linearlyDependent					= new ArrayList<>(dimensions);
		regressionCoeff						= new Hashtable<>(dimensions);
		ArrayList<Integer> active			= new ArrayList<>(dimensions);
		for (int d = 0; d < dimensions; d++)
		{
			ContStats series				= statistics.get(d);
			mean[d]							= series.getMean();
			double var						= series.getVar();
			variance[d]						= var;
			if (var < minVariance)			constant.add(d);
			else							active.add(d);
		}
		if (active.size() == 0)				return;
		if (randomBuild)					Collections.shuffle(active);
		
		// Determine clique size and count
		int k								= active.size();
		int cliqueSize						= Math.min(maxCliqueSize, (int)(cliqueRatio*n));
		cliqueSize							= Math.max(2, cliqueSize);
		double cliqueCountTemp				= (int)(k/(cliqueSize*(1 - sharePercent)));
		int cliqueCount						= (int)(Math.max(2.0, Math.ceil(cliqueCountTemp)));
		covIndexSize						= 2*cliqueSize;
		covIndex							= new Hashtable<>(covIndexSize);
		covQueue							= new ArrayBlockingQueue<>(covIndexSize);
		
		// Initialize clique arrays
		ArrayList<Clique> cliques			= new ArrayList<>(cliqueCount);
		int variable						= active.get(0);
		cliques.add(new Clique(variable, false, this));
		ArrayList<Clique> univarCliques		= new ArrayList<>(cliqueCount);
		int queueSize						= Math.max(1, k - cliqueCount);
		ArrayBlockingQueue<Integer> queue	= new ArrayBlockingQueue<>(queueSize);
		boolean firstCycle					= true;
		
		// Add variables
		this.samples						= samples;
		int d								= 0;
		while (true)
		{
			// Advance cycle and select variable
			if (firstCycle)
			{
				d++;
				if (d >= active.size())
				{
					firstCycle				= false;
					while (univarCliques.size() > 0)
					{
						Clique clique		= univarCliques.get(0);
						if (cliques.size() < cliqueCount)
							cliques.add(clique);
						else
							queue.offer(clique.getInitialVariable());
						univarCliques.remove(clique);
					}
				}
			}
			if (firstCycle)
				variable					= active.get(d);
			else
				if (queue.size() > 0)
					variable				= queue.poll();
			if (!firstCycle && queue.size() == 0)
				break;
			
			// Offer variable to cliques
			boolean isFree					= true;
			boolean isDetermined			= false;
			int selected					= -1;
			double selectedDelta			= Double.POSITIVE_INFINITY;			
			for (int c = 0; c < cliques.size(); c++)
			{
				Clique clique				= cliques.get(c);				
				clique.proposeVariable(variable);
				
				// Check if variable is free/independent or determined/fully-dependent
				double coeffDetermination	= clique.getCoeffDetermination();
				double explainedRatio		= clique.getExplainedVar()/variance[variable];
				
				if (coeffDetermination > freeR2Thresh && explainedRatio > freeVarThresh)
					isFree					= false;
				if ((coeffDetermination >= determR2Thresh && explainedRatio >= determVarThresh)
						|| clique.isDetermined())
				{
					linearlyDependent.add(variable);
					regressionCoeff.put(variable, clique.getRegressionCoeffs());
					isDetermined			= true;
					break;
				}
				
				// Add clique to selector if not full
				if (!isFree && clique.size() < cliqueSize)
				{
					double delta			= clique.getDeltaIndependenceIndex();
					if (delta < selectedDelta)
					{
						selected			= c;
						selectedDelta		= delta;
					}
				}
			}
			
			// Check if the variable is free/independent or determined/fully-dependent
			if (isDetermined)
				continue;
			if (isFree || (cliques.size() < cliqueCount && univarCliques.size() == 0))
			{			
				if (cliques.size() + univarCliques.size() < cliqueCount)
					univarCliques.add(new Clique(variable, false, this));
				else if (firstCycle)
					queue.offer(variable);
				else
					free.add(variable);
				continue;
			}
			
			// Add variable to clique
			Clique clique					= cliques.get(selected);
			double shared					= (double)clique.getShared()/(clique.size() + 1);
			double threshold				= sharePercent*
									(selected == 0 || selected == cliqueCount - 1 ? 0.5 : 1.0);
			boolean share					= shared < threshold;
			
			// Find clique to share variable with
			if (share)
			{
				Clique other;
				if (selected == cliqueCount - 1)		// Last clique
					other					= cliques.get(selected - 1);
				else if (selected < cliques.size() - 1)	// Intermediate clique
				{
					if (selected == 0)
						other				= cliques.get(1);
					else
					{
						Clique left			= cliques.get(selected - 1);
						Clique right		= cliques.get(selected + 1);
						double deltaL		= left.getDeltaIndependenceIndex();
						double deltaR		= right.getDeltaIndependenceIndex();
						other				= deltaL < deltaR ? left : right; 
					}
				}
				else									// Last non-univariate clique
				{
					selected				= -1;
					selectedDelta			= Double.POSITIVE_INFINITY;
					for (int c = 0; c < univarCliques.size(); c++)
					{
						Clique clique2		= univarCliques.get(c);
						clique2.proposeVariable(variable);
						double delta		= clique.getDeltaIndependenceIndex();
						if (delta < selectedDelta)
						{
							selected		= c;
							selectedDelta	= delta;
						}
					}
					if (selected >= 0)
					{
						other				= univarCliques.get(selected);
						cliques.add(other);
						univarCliques.remove(selected);
					}
					else
						other				= null;
				}
				if (other != null)
					other.commit(true);
				else
					share					= false;
			}
			clique.commit(share);
		}
		
		// Check for univariate cliques
		this.cliques						= new ArrayList<>(cliqueCount);
		for (Clique clique : cliques)
			if (clique.size() > 1)
				this.cliques.add(clique);
			else
				free.add(clique.getInitialVariable());
		for (Clique clique : univarCliques)
			free.add(clique.getInitialVariable());
		
		// Create variable index
		variableIndex						= new Hashtable<>(dimensions);
		for (Integer index : constant)
		{
			ArrayList<Integer> list			= new ArrayList<>();
			list.add(VAR_TYPE_CONSTANT);
			variableIndex.put(index, list);
		}
		for (Integer index : free)
		{
			ArrayList<Integer> list			= new ArrayList<>();
			list.add(VAR_TYPE_FREE);
			variableIndex.put(index, list);
		}
		for (Integer index : regressionCoeff.keySet())
		{
			ArrayList<Integer> list			= new ArrayList<>();
			list.add(VAR_TYPE_DETERMINED);
			variableIndex.put(index, list);
		}
		for (int c = 0; c < this.cliques.size(); c++)
		{
			for (Integer index : this.cliques.get(c).getVariables())
			{
				ArrayList<Integer> list		= variableIndex.get(index);
				if (list == null)
				{
					list					= new ArrayList<>();
					variableIndex.put(index, list);
				}
				list.add(c);
			}
		}
		
		long time							= System.currentTimeMillis() - start;
		int covariates						= mean.length - constant.size() - free.size()
												- regressionCoeff.size();
		String line							= "Created GGMLite (" + time + " ms) with "
				+ n + " samples; " + mean.length + " total dimensions: " + constant.size()
				+ " constant, " + free.size() + " free, " + regressionCoeff.size() + " dependent, "
				+ covariates + " covariates in " + this.cliques.size() + " cliques";
		if (this.cliques.size() > 0)
		{
			line							+= " of size = {";
			for (Clique clique : this.cliques)
				line						+= clique.size() + " ";
			line							= line.substring(0, line.length() - 1);
			line							+= "}";
		}
		System.out.println(line);
	}
	
	// --------------------------------------------------------------------------------------------
	// Methods
	// --------------------------------------------------------------------------------------------

	@Override
	public String getTypeString() 
	{
		return ID;
	}

	@Override
	public int getDimensionality() 
	{
		return mean.length;
	}

	/**
	 * @return {@link #mean}
	 */
	public double[] getMean() 
	{
		return mean;
	}

	/**
	 * @param mean {@link #mean}
	 */
	public void setMean(double[] mean)
	{
		this.mean			= mean;
		// TODO modify regression constant coefficients to match new mean 
	}
	
	@Override
	public double[][] getCovariance() 
	{
		// TODO Implement
		return null;
	}

	/**
	 * @return {@link #variance}
	 */
	public double[] getVariance()
	{
		return variance;
	}

	/**
	 * @return {@link #minVariance}
	 */
	public double getMinVariance()
	{
		return minVariance;
	}

	/**
	 * @param minVariance {@link #minVariance}
	 */
	public void setMinVariance(double minVariance) 
	{
		this.minVariance = minVariance;
	}
	
	/**
	 * @return {@link #constant}
	 */
	public ArrayList<Integer> getConstants()
	{
		return constant;
	}
	
	/**
	 * @return {@link #free}
	 */
	public ArrayList<Integer> getFree()
	{
		return free;
	}
	
	/**
	 * @return {@link #regressionCoeff}
	 */
	public Hashtable<Integer, ArrayList<PointID>> getRegCoeff()
	{
		return regressionCoeff;
	}

	/**
	 * Works as {@link #getpdf(double[])}
	 */
	public double getpdf(ArrayList<Double> x)
	{
		return Math.exp(getLogpdf(x));
	}
	
	/**
	 * Computes the value of the probability density function at the provided point. Returns 
	 * {@link java.lang.Double#NaN} if the distribution has not been correctly defined.
	 * @param x The point at which to evaluate the probability density
	 * @return The value of the probability density function at the provided point
	 */
	public double getpdf(double[] x)
	{
		return Math.exp(getLogpdf(x));
	}
	
	/**
	 * Works as {@link #getLogpdf(double[], boolean)}
	 */
	public double getLogpdf(ArrayList<Double> x)
	{
		double[] arrX	= new double[x.size()];
		for (int i = 0; i < x.size(); i++)
			arrX[i]		= x.get(i);
		return getLogpdf(arrX);
	}
	
	/**
	 * Computes the value of the natural logarithm of the probability density function at the 
	 * provided point. Returns {@link java.lang.Double#NaN} if the distribution has not been 
	 * correctly defined.
	 * @param x The point at which to evaluate the probability density
	 * @return The value of the natural logarithm of the probability density function at the 
	 * provided point
	 */
	public double getLogpdf(double[] x)
	{
		// TODO Implement
		return Double.NaN;
	}
	
	/**
	 * Computes the contribution to the log of the probability density from the cliques in the
	 * distribution at the provided point
	 * @param x	The point at which to evaluate the probability density
	 * @return	The contribution to the log of the probability density from the cliques in the
	 * distribution at the provided point
	 */
	public double getLogpdfCliques(double[] x)
	{
		int k			= mean.length;
		if (x.length != k)
			throw new IllegalArgumentException(String.format(ERR_SAMPLE_SIZE, x.length, k));
		double logpdf	= 0.0;
		for (Clique clique : cliques)
			logpdf		+= clique.getLogpdf(x);
		return logpdf;
	}
	
	/**
	 * Computes the Mahalanobis distance between a point and the mean of the distribution
	 * @param x	The point to compute the force with
	 * @return	The Mahalanobis distance between a point and the mean of the distribution
	 */
	public double getMahalanobisDistance(double[] x)
	{
		int k							= mean.length;
		if (x.length != k)
			throw new IllegalArgumentException(String.format(ERR_SAMPLE_SIZE, x.length, k));
		double sqrdDistance				= 0.0;
		
		// Free variables
		for (int f = 0; f < free.size(); f++)
		{
			int index					= free.get(f);
			double diff					= mean[index] - x[index];
			sqrdDistance				+= diff*diff/variance[index];
		}
		
		// Cliques
		for (int c = 0; c < cliques.size(); c++)
			sqrdDistance				+= cliques.get(c).getSqrdMahalanobisDistance(x);
		
		// Determined variables
		for (int d = 0; d < linearlyDependent.size(); d++)
		{
			int index					= linearlyDependent.get(d);
			ArrayList<PointID> coeffs	= regressionCoeff.get(index);
			double predicted			= 0.0;
			double stDev				= 0.0;
			for (PointID coeff : coeffs)
			{
				int index2				= coeff.x;
				if (index2 == -2)
					stDev				= coeff.y;
				else
					predicted			+= index2 == -1 ? coeff.y : coeff.y*x[index2];
			}
			double diff					= predicted - x[index];
			if (diff != 0.0)
			{
				stDev					= Math.max(stDev, Math.sqrt(minVariance));
				sqrdDistance			+= diff*diff/(stDev*stDev);
			}
		}
		
		return Math.sqrt(sqrdDistance);
	}
	
	/**
	 * Computes the Mahalanobis force of attraction between a point and the mean of the
	 * distribution. The Mahalanobis force is the inverse of the squared Mahalanobis distance.
	 * @param x	The point to compute the force with
	 * @return	The Mahalanobis force of attraction between a point and the mean of the 
	 * distribution
	 */
	public double getMahalanobisForce(double[] x)
	{
		int k							= mean.length;
		if (x.length != k)
			throw new IllegalArgumentException(String.format(ERR_SAMPLE_SIZE, x.length, k));
		double sqrdDistance				= 0.0;
		
		// Free variables
		for (int f = 0; f < free.size(); f++)
		{
			int index					= free.get(f);
			double diff					= mean[index] - x[index];
			if (diff != 0.0)
			{
				double var				= Math.max(variance[index], minVariance);
				sqrdDistance			+= diff*diff/var;
			}
		}
		
		// Cliques
		for (int c = 0; c < cliques.size(); c++)
			sqrdDistance				+= cliques.get(c).getSqrdMahalanobisDistance(x);
		
		// Determined variables
		for (int d = 0; d < linearlyDependent.size(); d++)
		{
			int index					= linearlyDependent.get(d);
			ArrayList<PointID> coeffs	= regressionCoeff.get(index);
			double predicted			= 0.0;
			double stDev				= 0.0;
			for (PointID coeff : coeffs)
			{
				int index2				= coeff.x;
				if (index2 == -2)
					stDev				= coeff.y;
				else
					predicted			+= index2 == -1 ? coeff.y : coeff.y*x[index2];
			}
			double diff					= predicted - x[index];
			if (diff != 0.0)
			{
				stDev					= Math.max(stDev, Math.sqrt(minVariance));
				sqrdDistance			+= diff*diff/(stDev*stDev);
			}
		}
		
		return 1/sqrdDistance;
	}
	
	/**
	 * @param x	The point to compute the distance to
	 * @return	The squared Mahalanobis distance between a point and the mean of the distribution
	 */
	public double getSqrdMahalanobisDistCliques(double[] x)
	{
		int k			= mean.length;
		if (x.length != k)
			throw new IllegalArgumentException(String.format(ERR_SAMPLE_SIZE, x.length, k));
		double distance	= 0.0;
		for (Clique clique : cliques)
			distance	+= clique.getSqrdMahalanobisDistance(x);
		return distance;
	}
	
	// TODO Return regression coefficients
	
	/**
	 * Marginalizes the distribution of one of the dimensions. That is, creates a reduced 
	 * distribution of only one of the original dimensions.
	 * <p>Time complexity: <i>O(number of samples)</i>
	 * <br>Space complexity: <i>O(number of samples)</i>
	 * @param dimension The index of the dimension to marginalize
	 * @return The marginalized normal distribution
	 */
	public Normal marginalize(int dimension)
	{
		// TODO Implement
		return null;
	}
	
	/**
	 * Marginalizes the distribution of a set of variables. That is, creates a reduced distribution 
	 * with only a set of the original dimensions.
	 * @param toActivate A set containing the indices of the dimensions to marginalize
	 * @return The marginalized multivariate normal distribution. <code>null</code> if the 
	 * activation parameter contained no dimensions to marginalize.
	 */
	public GGMLite marginalize(ArrayList<Integer> toActivate)
	{
		// TODO Implement
		return null;
	}
	
	@Override
	public MultiContProbDist conditional(ArrayList<Double> x) 
	{
		// TODO Implement
		return null;
	}
	
	/**
	 * Generates a random vector from the multivariate normal distribution
	 * @return A random vector from the multivariate normal distribution
	 */
	public double[] sample()
	{
		int k							= mean.length;
		double[] sample					= new double[k];
		
		// Sample values from cliques
		Hashtable<Integer, Double> vals	= new Hashtable<>(k);
		for (Clique clique : cliques)
			clique.sample(vals);
		for (Integer index : vals.keySet())
			sample[index]				= vals.get(index);
		
		// Add determined/dependent values
		for (Integer index2 : linearlyDependent)
		{
			ArrayList<PointID> reg		= regressionCoeff.get(index2);
			double value				= 0.0;
			for (PointID point : reg)
			{
				int index				= point.x;
				switch (index)
				{
					case -1:			value += point.y;									break;
					case -2:			value += Normal.sample(0.0, Math.sqrt(point.y));	break;
					default:			value += sample[index]*point.y;
				}
			}
			sample[index2]				= value;
		}
		
		// Add values for constant and free/independent variables
		for (Integer index : constant)
			sample[index]				= mean[index];
		for (Integer index : free)
			sample[index]				= Normal.sample(mean[index], Math.sqrt(variance[index]));
		
		return sample;
	}
	
	@Override
	public double[][] sampleMultiple(int count) 
	{
		int k				= mean.length;
		double[][] samples	= new double[count][k];
		for (int i = 0; i < count; i++)
			samples[i]		= sample();
		return samples;
	}

	@Override
	public double getMean(int variable)
	{
		return mean[variable];
	}

	@Override
	public double getCovariance(int variable1, int variable2)
	{
		if (variable1 == variable2)
			return variance[variable1];
		
		if (samples != null)
		{
			Point2I key						= new Point2I(variable1, variable2);
			Double stored					= covIndex.get(key);
			if (stored == null)
			{
				key							= new Point2I(variable2, variable1);
				stored						= covIndex.get(key);
			}
			if (stored != null)
				return stored;
			
			// Compute covariance
			ContPairedSeries series			= new ContPairedSeries(true);
			for (ContMultiSample sample : samples)
			{
				ArrayList<Double> values	= sample.getValues();
				series.addPair(values.get(variable1), values.get(variable2), sample.getWeight());
			}
			double covariance				= series.getCov();
			
			// Add value and return
			while (covIndex.size() >= covIndexSize - 1)
			{
				Point2I key2				= covQueue.poll();
				covIndex.remove(key2);
			}
			key								= new Point2I(variable1, variable2);
			covIndex.put(key, covariance);
			covQueue.offer(key);
			return covariance;
		}
		
		// TODO Implement (when samples are not available)
		return Double.NaN;
	}

	@Override
	public double getCoefficientDetermination(int variable, ArrayList<PointID> regCoeff) 
	{
		double resSumSquares			= 0;
		double totSumSquares			= 0;
		double varMean					= mean[variable];
		for (int s = 0; s < samples.size(); s++)
		{
			ContMultiSample sample		= samples.get(s);
			ArrayList<Double> values	= sample.getValues();
			double weight				= sample.getWeight();
			double predicted			= 0.0;
			for (PointID term : regCoeff)
			{
				int index				= term.x;
				if (index == -1)
					predicted			+= term.y;
				else
				{
					double value		= values.get(index);
					predicted			+= term.y*value;
				}
			}
			double value				= values.get(variable);
			double residual				= value - predicted;
			double deviation			= value - varMean;
			resSumSquares				= weight*residual*residual;
			totSumSquares				= weight*deviation*deviation;
		}
		return 1 - resSumSquares/totSumSquares;
	}
	
}
