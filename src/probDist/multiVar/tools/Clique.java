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
import java.util.Hashtable;

import probDist.multiVar.MultiVarNormal;
import utilities.MatUtil;
import utilities.geom.PointID;

/**
 * A group of interrelated variables, whose dependencies are encoded through a covariance matrix 
 * @author Felipe Hernández
 */
public class Clique
{

	// --------------------------------------------------------------------------------------------
	// Constants
	// --------------------------------------------------------------------------------------------
	
	/**
	 * Error message: negative variable index
	 */
	public final static String ERR_INVALID_INDEX = "Variables' indices should be positive (the "
													+ "proposed one is %1$d)";
	
	/**
	 * Error message: variable already in clique
	 */	
	public final static String ERR_VARIABLE_PRESENT = "Variable %1$d is already in the clique";
	
	/**
	 * Error message: no variable has been proposed
	 */
	public final static String ERR_NO_VARIABLE = "No variable has been proposed";

	/**
	 * Error message: invalid variable was proposed
	 */
	public final static String ERR_INVAILD_VARIABLE = "The proposed variable (%1$d) would make the"
			+ " system non positive-definite (it is fully determined by the existing variables)";
	
	/**
	 * Error message: invalid vector size
	 */
	public final static String ERR_VECTOR_SIZE = "The provided vector size is %1$f but it should"
													+ " be %2$f";

	/**
	 * The minimum value for the determinant of a matrix to consider it invertible
	 */
	public final static double INVERTIBLE_THRESHOLD = 1E-200;
	
	// --------------------------------------------------------------------------------------------
	// Attributes
	// --------------------------------------------------------------------------------------------

	/**
	 * The positions in the {@link #matrix} of each of the variables
	 */
	private Hashtable<Integer, Integer> varsIndices;
	
	/**
	 * The indices of the variables for each of the positions in {@link #matrix}
	 */
	private ArrayList<Integer> indicesVars;
	
	/**
	 * The number of variables in the clique that are shared with other cliques
	 */
	private int shared;
	
	/**
	 * Covariance matrix of the variables in the clique
	 */
	private double[][] matrix;
	
	/**
	 * The Cholesky factor <i>L</i> of the {@link #matrix} (<i>C</i>): <i>C = L*L^T</i>
	 */
	private double[][] cholesky;
	
	/**
	 * The determinant of the {@link #matrix}
	 */
	private double determinant;
	
	/**
	 * The independence index of the clique, defined as the determinant of the {@link #matrix}
	 * multiplied by the product of the variance values of each variable in the clique
	 */
	private double independenceIndex;
	
	/**
	 * The index of the most-recently proposed variable
	 */
	private int proposed;
	
	/**
	 * The coefficient of the determination <i>R<sup>2</sup></i> of the {@link #proposed} variable
	 * using the regression encoded in the {@link #regressionCoeffs}.
	 */
	private double coeffDetermination;
	
	/**
	 * The variance of the {@link #proposed} variable as estimated using the
	 * {@link #regressionCoeffs}
	 */
	private double explainedVar;
	
	/**
	 * The list of regression coefficients to estimate the {@link #proposed} variable from those in
	 * the clique
	 */
	private ArrayList<PointID> regressionCoeffs;
	
	/**
	 * Extended covariance matrix of the variables in the clique (including the {@link #proposed}
	 * one)
	 */
	private double[][] matrix2;
	
	/**
	 * The Cholesky factor <i>L</i> of {@link #matrix2} (<i>C2</i>): <i>C2 = L*L^T</i>
	 */
	private double[][] cholesky2;
	
	/**
	 * The determinant of {@link #matrix2}
	 */
	private double determinant2;
	
	/**
	 * The independence index of the clique, defined as the determinant of {@link #matrix2}
	 * multiplied by the product of the variance values of each variable in the clique (including
	 * the {@link #proposed} one)
	 */
	private double independenceIndex2;
	
	/**
	 * Instance of the object that computes statistics of variables 
	 */
	private StatsCalculator statsCalculator;
	
	// --------------------------------------------------------------------------------------------
	// Constructor
	// --------------------------------------------------------------------------------------------
	
	/**
	 * Initializes a clique with an initial variable
	 * @param index				The index of the initial variable
	 * @param shared			True if the variable exists in another clique
	 * @param statsCalculator	{@link #statsCalculator}
	 */
	public Clique(int index, boolean shared, StatsCalculator statsCalculator)
	{
		if (index < 0) throw new IllegalArgumentException(String.format(ERR_INVALID_INDEX, index));
		
		varsIndices				= new Hashtable<>();
		varsIndices.put(index, 0);
		indicesVars				= new ArrayList<>();
		indicesVars.add(index);
		this.shared				= shared ? 1 : 0;
		matrix					= new double[1][1];
		double var				= statsCalculator.getCovariance(index, index);
		matrix[0][0]			= var;
		cholesky				= new double[1][1];
		cholesky[0][0]			= Math.sqrt(var);
		determinant				= var;
		independenceIndex		= 1.0;
		proposed				= -1;
		coeffDetermination		= Double.NaN;
		explainedVar			= Double.NaN;
		regressionCoeffs		= null;
		this.statsCalculator	= statsCalculator;
	}
	
	// --------------------------------------------------------------------------------------------
	// Methods
	// --------------------------------------------------------------------------------------------
	
	/**
	 * @return The initial variable in the clique
	 */
	public int getInitialVariable()
	{
		return indicesVars.get(0);
	}
	
	/**
	 * @return The number of variables currently in the clique
	 */
	public int size()
	{
		return varsIndices.size();
	}
	
	/**
	 * @param variable The index of the variable to look for
	 * @return True if the variable is already in the clique
	 */
	public boolean contains(int variable)
	{
		return varsIndices.containsKey(variable);
	}
	
	/**
	 * @return A list with the indices of the variables in the clique
	 */
	public ArrayList<Integer> getVariables()
	{
		return indicesVars;
	}
	
	/**
	 * @return {@link #shared}
	 */
	public int getShared()
	{
		return shared;
	}
	
	/**
	 * @return {@link #determinant}
	 */
	public double getDeterminant()
	{
		return determinant;
	}
	
	/**
	 * Evaluates the possibility of adding a new variable to the clique: it could be found 
	 * unrelated to those already in the clique; it could be found fully dependent/determined by
	 * those in the clique; or it could be found somewhere in between
	 * @param index				The index of the variable being proposed
	 */
	public void proposeVariable(int index)
	{
		if (index < 0)
			throw new IllegalArgumentException(String.format(ERR_INVALID_INDEX, index));
		if (varsIndices.containsKey(index))
			throw new IllegalArgumentException(String.format(ERR_VARIABLE_PRESENT, index));
		proposed					= index;
		
		// Prepare extended matrix
		int size					= varsIndices.size();
		int size2					= size + 1;
		matrix2						= new double[size2][size2];
		for		(int i = 0; i < size; i++)
			for	(int j = 0; j < size; j++)
				matrix2[i][j]		= matrix[i][j];
		
		// Compute covariance values
		double[] covs				= new double[size];
		for (int v = 0; v < size; v++)
		{
			int var2				= indicesVars.get(v);
			double cov				= statsCalculator.getCovariance(index, var2);
			covs[v]					= cov;
			matrix2[v][size]		= cov;
			matrix2[size][v]		= cov;
		}
		double variance				= statsCalculator.getCovariance(index, index);
		matrix2[size][size]			= variance;
		
		// Compute determinant
		determinant2				= 0.0;
		try
		{
			cholesky2				= MatUtil.choleskyStepWise(matrix2, cholesky);
			double detL				= MatUtil.getDeterminantTriangular(cholesky2);
			determinant2			= detL*detL;
		} catch (Exception e) {}
		
		// Determine regression coefficients
		double[] coeffs				= MatUtil.luEvaluate(cholesky, covs);
		double mean					= statsCalculator.getMean(index);
		double[] means				= new double[size];
		for (int c = 0; c < size; c++)
			means[c]				= statsCalculator.getMean(indicesVars.get(c));
		double combination			= MatUtil.dotProduct(coeffs, means);
		double constant				= mean - combination;
		
		// Compute dependence percentage
		explainedVar				= 0.0;
		for 	(int c1 = 0; c1 < size; c1++)
		{
			double coeff1			= coeffs[c1];
			explainedVar			+= coeff1*coeff1*matrix[c1][c1];
			for	(int c2 = c1 + 1; c2 < size; c2++)
			{
				double coeff2		= coeffs[c2];
				explainedVar		+= 2*coeff1*coeff2*matrix[c1][c2];
			}
		}
		double selfVar				= Math.max(0.0, variance - explainedVar);
		
		// Store regression coefficients
		regressionCoeffs			= new ArrayList<>(2 + size);
		regressionCoeffs.add(new PointID(-1, constant	));
		for (int c = 0; c < size; c++)
			if (coeffs[c] != 0.0)
				regressionCoeffs.add(new PointID(indicesVars.get(c), coeffs[c]));
		
		// Determine variable type
		coeffDetermination			= statsCalculator.getCoefficientDetermination(proposed, 
																				regressionCoeffs);
		regressionCoeffs.add(new PointID(-2, selfVar	));

		// Compute change in independence index
		double sum					= Math.log(determinant2);
		for (int c = 0; c < size2; c++)
			sum						-= Math.log(matrix2[c][c]);
		independenceIndex2			= Math.exp(sum);
	}
	
	/**
	 * @return <code>true</code> if {@link #determinant2} is smaller than 
	 * {@link #INVERTIBLE_THRESHOLD}, that is, if the offered variable is fully determined by those
	 * already in the clique. If it is, committing it to the clique would make the system non
	 * positive-definite. 
	 */
	public boolean isDetermined()
	{
		return determinant2 < INVERTIBLE_THRESHOLD;
	}
	
	/**
	 * @return {@link #coeffDetermination}
	 */
	public double getCoeffDetermination()
	{
		if (proposed == -1)
			throw new RuntimeException(ERR_NO_VARIABLE);
		return coeffDetermination;
	}
	
	/**
	 * @return {@link #explainedVar}
	 */
	public double getExplainedVar()
	{
		if (proposed == -1)
			throw new RuntimeException(ERR_NO_VARIABLE);
		return explainedVar;
	}
	
	/**
	 * @return {@link #regressionCoeffs}
	 */
	public ArrayList<PointID> getRegressionCoeffs()
	{
		if (proposed == -1)
			throw new RuntimeException(ERR_NO_VARIABLE);
		return regressionCoeffs;
	}
	
	/**
	 * @return The difference between {@link #independenceIndex2} and {@link #independenceIndex}
	 */
	public double getDeltaIndependenceIndex()
	{
		if (proposed == -1)
			throw new RuntimeException(ERR_NO_VARIABLE);
		return independenceIndex2 - independenceIndex;
	}
	
	/**
	 * Add the variable previously proposed by calling 
	 * {@link #proposeVariable(int, double, double, double)} to the clique
	 * @param shared True if the variable is to also be added to another clique
	 */
	public void commit(boolean shared)
	{
		if (proposed == -1 || Double.isNaN(determinant2))
			throw new RuntimeException(ERR_NO_VARIABLE);
		if (determinant2 < INVERTIBLE_THRESHOLD)
			throw new RuntimeException(String.format(ERR_INVAILD_VARIABLE, proposed));
		
		int size				= size();
		varsIndices.put(proposed, size);
		indicesVars.add(proposed);
		if (shared) this.shared++;
		matrix					= matrix2;
		cholesky				= cholesky2;
		determinant				= determinant2;
		independenceIndex		= independenceIndex2;
		proposed				= -1;
	}
	
	/**
	 * Computes the value of the probability density function at the provided point
	 * @param point	The point at which to evaluate the probability density including all values
	 * 				(even those that do not make part of the clique)
	 * @return		The value of the probability density function at the provided point
	 */
	public double getLogpdf(double[] point)
	{
		// Prepare vector of deviations
		int dimensions			= indicesVars.size();
		double[] deviations		= new double[dimensions];
		for (int v = 0; v < dimensions; v++)
		{
			int index			= indicesVars.get(v);
			deviations[v]		= point[index] - statsCalculator.getMean(index);
		}
		
		// Compute log probability density
		double[] y				= MatUtil.solveLower(cholesky, deviations);
		double sqrdDist			= 0.0;
		for (int v = 0; v < dimensions; v++)
			sqrdDist			+= y[v]*y[v];
		return -0.5*sqrdDist - Math.log(Math.sqrt(Math.pow(2*Math.PI, dimensions)*determinant));
	}
	
	/**
	 * Computes the squared Mahalanobis distance between a point and the mean of the clique
	 * @param point	The all the values of the point (even those that do not make part of the 
	 * 				clique)
	 * @return		The squared Mahalanobis distance between a point and the mean of the clique
	 */
	public double getSqrdMahalanobisDistance(double[] point)
	{		
		// Prepare vector of deviations
		int dimensions			= indicesVars.size();
		double[] deviations		= new double[dimensions];
		for (int v = 0; v < dimensions; v++)
		{
			int index			= indicesVars.get(v);
			deviations[v]		= point[index] - statsCalculator.getMean(index);
		}
		
		// Compute distance
		double[] y				= MatUtil.solveLower(cholesky, deviations);
		double sqrdDist			= 0.0;
		for (int v = 0; v < dimensions; v++)
			sqrdDist			+= y[v]*y[v];
		return sqrdDist;
	}

	/**
	 * @return A random sample of the variables in the clique
	 */
	public double[] sample()
	{
		int dimensions			= indicesVars.size();
		double[] mean			= new double[dimensions];
		for (int v = 0; v < dimensions; v++)
			mean[v]				= statsCalculator.getMean(indicesVars.get(v));
		MultiVarNormal normal	= new MultiVarNormal(mean, matrix);
		normal.markCovarianceAsInvertible();
		return normal.sample();
	}
	
	/**
	 * Creates a random sample of the variables in the clique conditioned on the known values
	 * provided. That is, a conditional distribution is created given the provided known values and
	 * then that distribution is sampled. The new sampled values are added to the provided hash 
	 * table of known values.
	 * @param knownValues 	A table with the known values for each variable index. The new samples
	 * 						values are added here.
	 */
	public void sample(Hashtable<Integer, Double> knownValues)
	{
		int dimensions					= indicesVars.size();		
		// Determine number of known/unknown variables
		boolean areThereKnowns			= true;
		ArrayList<Integer> knownVars	= null;
		ArrayList<Integer> unknownVars	= null;
		if (knownValues.size() == 0)
			areThereKnowns				= false;
		else
		{
			knownVars					= new ArrayList<>(dimensions);
			unknownVars					= new ArrayList<>(dimensions);
			for (int v = 0; v < dimensions; v++)
				if (knownValues.containsKey(indicesVars.get(v)))
					knownVars.add(v);
				else
					unknownVars.add(v);
			if (knownVars.size() == 0)
				areThereKnowns			= false;
		}
		
		// Sample without conditioning if there are no known variables
		if (!areThereKnowns)
		{
			double[] sample				= sample();
			for (int v = 0; v < dimensions; v++)
			{
				int variable			= indicesVars.get(v);
				knownValues.put(variable, sample[v]);
			}
			return;
		}
		
		// Prepare re-organized matrices for conditioning
		int nu							= unknownVars.size();
		int nk							= knownVars.size();
		double[] miu1					= new double[nu];
		double[] miu2					= new double[nk];
		double[][] sigma11				= new double[nu][nu];
		double[][] sigma12				= new double[nu][nk];
		double[][] chol22				= new double[nk][nk];	// Cholesky of sigma22
		for (int u = 0; u < nu; u++)
		{
			int index					= unknownVars.get(u);
			int var						= indicesVars.get(index);
			miu1[u]						= statsCalculator.getMean(var);
			sigma11[u][u]				= matrix[index][index];
			for (int u2 = 0; u2 < u; u2++)
			{
				int index2				= unknownVars.get(u2);
				double cov				= matrix[index][index2];
				sigma11[u][u2]			= cov;
				sigma11[u2][u]			= cov;
			}
		}
		for (int k = 0; k < nk; k++)
		{
			int index					= knownVars.get(k);
			int var						= indicesVars.get(index);
			miu2[k]						= statsCalculator.getMean(var);
			for (int k2 = k; k2 < nk; k2++)
			{
				int index2				= knownVars.get(k2);
				double chol				= cholesky[index2][index];
				chol22[k2][k]			= chol;
			}
			for (int u = 0; u < nu; u++)
			{
				int index2				= unknownVars.get(u);
				double cov				= matrix[index][index2];
				sigma12[u][k]			= cov;
			}
		}
		
		// Invert the covariance of the known variables (sigma22)
		double[][] L_inv				= MatUtil.invertLowerTriangular(chol22);
		double[][] L_inv_T				= MatUtil.transpose(L_inv);
		double[][] sigma22_inv			= MatUtil.multiply(L_inv_T, L_inv);
		
		// Determine conditional distribution
		double[] devs					= new double[nk];
		for (int k = 0; k < nk; k++)
		{
			int var						= indicesVars.get(knownVars.get(k));
			devs[k]						= knownValues.get(var) - miu2[k];
		}
		double[][] temp					= MatUtil.multiply(sigma12, sigma22_inv);
		double[] tempVec				= MatUtil.multiply(temp, devs);
		double[] condMean				= new double[nu];
		for (int u = 0; u < nu; u++)
			condMean[u]					= miu1[u] + tempVec[u];
		temp							= MatUtil.multiply(temp, MatUtil.transpose(sigma12));
		double[][] condCovs				= new double[nu][nu];
		for 	(int u1 = 0; u1 < nu; u1++)
			for	(int u2 = 0; u2 < nu; u2++)
				condCovs[u1][u2]		= sigma11[u1][u2] - temp[u1][u2];
		MultiVarNormal conditional		= new MultiVarNormal(condMean, condCovs);
		
		// Create sample and store values
		double[] sample					= conditional.sample();
		for (int u = 0; u < nu; u++)
		{
			int var						= indicesVars.get(unknownVars.get(u));
			knownValues.put(var, sample[u]);
		}
	}
	
}
