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

import utilities.geom.PointID;

/**
 * Computes statistics of variables given their indices
 * @author Felipe Hernández
 */
public interface StatsCalculator
{

	/**
	 * @param variable The index of the variable
	 * @return The mean of the indicated variable
	 */
	public double getMean(int variable);
	
	/**
	 * @param variable1	The index of the first variable
	 * @param variable2	The index of the second variable
	 * @return The covariance between the indicated variables
	 */
	public double getCovariance(int variable1, int variable2);
	
	/**
	 * Computes the coefficient of determination <i>R<sup>2</sup></i> of a regression
	 * @param variable	The variable being estimated
	 * @param regCoeff	The regression coefficients: the list contains tuples with the index of the
	 * 					explanatory variables and the corresponding coefficient. The -1 index in 
	 * 					the tuples represents the constant coefficient.
	 * @return The coefficient of determination <i>R<sup>2</sup></i>
	 */
	public double getCoefficientDetermination(int variable, ArrayList<PointID> regCoeff);
	
}
