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

package probDist.plot;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

import javax.imageio.ImageIO;

import probDist.ContProbDist;

public class ProbTimeSeries
{

	private static final double DRAW_THRESHOLD = 0.005;

	public static void plotTimeSeries(ArrayList<ContProbDist> series, double minY, double maxY,
			boolean logarithmicScale, int pixelsY, Color background, ArrayList<Color> colorRamp,
			String file) throws IOException
	{
		int width					= series.size();
		int height					= pixelsY;
		int imageType				= BufferedImage.TYPE_4BYTE_ABGR;
		BufferedImage image			= new BufferedImage(width, height, imageType);
		
		Graphics2D graph			= image.createGraphics();
		graph.setPaint(background);
		graph.fillRect(0, 0, image.getWidth(), image.getHeight());		
		
		int x						= 0;
		for (ContProbDist dist : series)
		{
			// Obtain density values
			double max				= 0.0;
			ArrayList<Double> pdfs	= new ArrayList<>(pixelsY);
			for (int y = 0; y < pixelsY; y++)
			{
				double base			= (double)y/(pixelsY - 1);
				double val			= logarithmicScale
									? Math.pow(10.0, Math.log10(minY) + base*Math.log10(maxY/minY))
									: minY + base*(maxY - minY);
				double pdf			= dist.getpdf(val);
				max					= pdf > max ? pdf : max;
				pdfs.add(pdf);
			}
			
			// Assign colors
			for (int y = 0; y < pixelsY; y++)
			{
				double pdf			= pdfs.get(y);
				double fraction		= pdf/max;
				if (fraction > DRAW_THRESHOLD)
				{
					Color color		= getColor(colorRamp, fraction);
					image.setRGB(x, pixelsY - y - 1, color.getRGB());
				}
			}
			x++;
		}
		
		File outputfile			= new File(file);
		ImageIO.write(image, "png", outputfile);
	}
	
	private static Color getColor(ArrayList<Color> colorRamp, double fraction)
	{
		int colorCount			= colorRamp.size();
		double scaled			= fraction*(colorCount - 1);
		int color1				= (int)Math.floor(	scaled);
		int color2				= (int)Math.ceil(	scaled);
		if (color1 == color2)
			return colorRamp.get(color1);
		
		double innerFraction	= scaled - color1;
		return gradient(colorRamp.get(color1), colorRamp.get(color2), innerFraction);
	}
	
	private static Color gradient(Color color1, Color color2, double fraction)
	{
		int red			= (int)(color1.getRed()*	(1 - fraction) + color2.getRed()*	fraction);
		int green		= (int)(color1.getGreen()*	(1 - fraction) + color2.getGreen()*	fraction);
		int blue		= (int)(color1.getBlue()*	(1 - fraction) + color2.getBlue()*	fraction);
		int alpha		= (int)(color1.getAlpha()*	(1 - fraction) + color2.getAlpha()*	fraction);
		return new Color(red, green, blue, alpha);
	}

}
