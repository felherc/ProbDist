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

package probDist.multiVar.tools;

import java.util.ArrayList;

import probDist.multiVar.GGMLite;

/**
 * Facilitates the creation of instances of type {@link probDist.multiVar.GGMLite} by allowing to
 * specify creation parameters 
 * @author Felipe Hernández
 */
public class GGMLiteCreator
{

	// --------------------------------------------------------------------------------------------
	// Attributes
	// --------------------------------------------------------------------------------------------
	
	/**
	 * The ratio of the number of samples <i>n</i> to limit the size of the cliques (since the rank 
	 * of covariance matrices is limited by <i>n</i>). Should be larger than zero.
	 */
	private double cliqueRatio;
	
	/**
	 * A hard limit on the size of the cliques which controls the size of the covariance matrices
	 * used to relate co-dependent variables
	 */
	private int maxCliqueSize;
	
	/**
	 * The percentage of variables to be shared by cliques (and thus to maintain their 
	 * interconnection)
	 */
	private double sharePercent;
	
	/**
	 * The maximum value for the coefficient of determination <i>R<sup>2</sup></i> of the 
	 * regression based on other variables to consider a variable free/independent
	 */
	private double freeR2Thresh;
	
	/**
	 * The maximum value for the percentage of variance explained by the regression to consider a 
	 * variable free/independent
	 */
	private double freeVarThresh;
	
	/**
	 * The minimum value for the coefficient of determination <i>R<sup>2</sup></i> of the 
	 * regression based on other variables to consider a variable fully determined/dependent
	 */
	private double determR2Thresh;
	
	/**
	 * The minimum value for the percentage of variance explained by the regression to consider a
	 * variable fully determined/dependent
	 */
	private double determVarThresh;
	
	/**
	 * True if the clique array should be built by adding variables in a random order
	 */
	private boolean randomBuild;
	
	/**
	 * Minimum allowed value for the covariance matrix before it is considered zero for likelihood
	 * and sampling computation purposes. Also used as the minimum value to assume that a
	 * determinant is practically zero (the matrix is singular), and that a mean value for a
	 * dimension of the distribution is practically zero.
	 */
	private double minVariance;
	
	// --------------------------------------------------------------------------------------------
	// Constructors
	// --------------------------------------------------------------------------------------------
	
	/**
	 * Initializes the creator with default attribute assignments
	 */
	public GGMLiteCreator()
	{
		this.cliqueRatio		= GGMLite.DEF_CLIQUE_RATIO;
		this.maxCliqueSize		= GGMLite.DEF_MAX_CLIQUE_SIZE;
		this.sharePercent		= GGMLite.DEF_SHARE_PERCENT;
		this.freeR2Thresh		= GGMLite.DEF_FREE_R2_THRESH;
		this.freeVarThresh		= GGMLite.DEF_FREE_VAR_THRESH;
		this.determR2Thresh		= GGMLite.DEF_DETERM_R2_THRESH;
		this.determVarThresh	= GGMLite.DEF_DETERM_VAR_THRESH;
		this.randomBuild		= GGMLite.DEF_RANDOM_BUILD;
		this.minVariance		= GGMLite.DEF_MIN_VARIANCE;
	}

	/**
	 * @param cliqueRatio		{@link #cliqueRatio}
	 * @param maxCliqueSize		{@link #maxCliqueSize}
	 * @param sharePercent		{@link #sharePercent}
	 * @param freeR2Thresh		{@link #freeR2Thresh}
	 * @param freeVarThresh		{@link #freeVarThresh}
	 * @param determR2Thresh	{@link #determR2Thresh}
	 * @param determVarThresh	{@link #determVarThresh}
	 * @param randomBuild		{@link #randomBuild}
	 * @param minVariance		{@link #minVariance}
	 */
	public GGMLiteCreator(double cliqueRatio, int maxCliqueSize, double sharePercent, 
			double freeR2Thresh, double freeVarThresh, double determR2Thresh,
			double determVarThresh, boolean randomBuild, double minVariance)
	{
		this.cliqueRatio		= 		Math.max(0.0,	cliqueRatio			);
		this.maxCliqueSize		= (int)	Math.max(2,		maxCliqueSize		);
		this.sharePercent		= 		Math.max(0.0,	sharePercent		);
		this.sharePercent		= 		Math.min(1.0,	this.sharePercent	);
		this.freeR2Thresh		= 		Math.min(1.0,	freeR2Thresh		);
		this.freeVarThresh		= 						freeVarThresh		;
		this.determR2Thresh		=		Math.min(1.0,	determR2Thresh		);
		this.determVarThresh	= 						determVarThresh		;
		this.randomBuild		= 						randomBuild			;
		this.minVariance		= 		Math.max(0.0,	minVariance			);
	}
	
	// --------------------------------------------------------------------------------------------
	// Methods
	// --------------------------------------------------------------------------------------------

	/**
	 * @return {@link #cliqueRatio}
	 */
	public double getCliqueRatio()
	{
		return cliqueRatio;
	}

	/**
	 * @param cliqueRatio {@link #cliqueRatio}
	 */
	public void setCliqueRatio(double cliqueRatio)
	{
		this.cliqueRatio = Math.max(0.0, cliqueRatio);
	}

	/**
	 * @return {@link #maxCliqueSize}
	 */
	public int getMaxCliqueSize()
	{
		return maxCliqueSize;
	}

	/**
	 * @param maxCliqueSize {@link #maxCliqueSize}
	 */
	public void setMaxCliqueSize(int maxCliqueSize)
	{
		this.maxCliqueSize = (int) Math.max(2, maxCliqueSize);
	}

	/**
	 * @return {@link #sharePercent}
	 */
	public double getSharePercent()
	{
		return sharePercent;
	}

	/**
	 * @param sharePercent {@link #sharePercent}
	 */
	public void setSharePercent(double sharePercent)
	{
		this.sharePercent = Math.max(0.0, sharePercent		);
		this.sharePercent = Math.min(1.0, this.sharePercent	);
	}

	/**
	 * @return {@link #freeR2Thresh}
	 */
	public double getFreeR2Thresh()
	{
		return freeR2Thresh;
	}

	/**
	 * @param freeR2Thresh {@link #freeR2Thresh}
	 */
	public void setFreeR2Thresh(double freeR2Thresh)
	{
		this.freeR2Thresh = Math.min(1.0, freeR2Thresh);
	}

	/**
	 * @return {@link #freeVarThresh}
	 */
	public double getFreeVarThresh()
	{
		return freeVarThresh;
	}

	/**
	 * @param freeVarThresh {@link #freeVarThresh}
	 */
	public void setFreeVarThresh(double freeVarThresh)
	{
		this.freeVarThresh = freeVarThresh;
	}

	/**
	 * @return {@link #determR2Thresh}
	 */
	public double getDetermR2Thresh()
	{
		return determR2Thresh;
	}

	/**
	 * @param determR2Thresh {@link #determR2Thresh}
	 */
	public void setDetermR2Thresh(double determR2Thresh)
	{
		this.determR2Thresh = Math.min(1.0, determR2Thresh);
	}

	/**
	 * @return {@link #determVarThresh}
	 */
	public double getDetermVarThresh()
	{
		return determVarThresh;
	}

	/**
	 * @param determVarThresh {@link #determVarThresh}
	 */
	public void setDetermVarThresh(double determVarThresh)
	{
		this.determVarThresh = determVarThresh;
	}

	/**
	 * @return {@link #randomBuild}
	 */
	public boolean getRandomBuild()
	{
		return randomBuild;
	}

	/**
	 * @param randomBuild {@link #randomBuild}
	 */
	public void setRandomBuild(boolean randomBuild)
	{
		this.randomBuild = randomBuild;
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
		this.minVariance = Math.max(0.0, minVariance);
	}
	
	/**
	 * @param samples The samples of the underlying distribution to model. Samples can have missing
	 * values, which should be set to {@link java.lang.Double#NaN}.
	 * @return The fitted sparse Gaussian Graphical Model
	 */
	public GGMLite create(ArrayList<ContMultiSample> samples)
	{
		return new GGMLite(samples, cliqueRatio, maxCliqueSize, sharePercent, freeR2Thresh,
						freeVarThresh, determR2Thresh, determVarThresh, randomBuild, minVariance);
	}
	
}
