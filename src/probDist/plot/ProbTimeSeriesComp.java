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

public class ProbTimeSeriesComp
{

	public static final double DRAW_THRESHOLD = 0.005;
	public static final double LOGISTIC_STEEPNESS = 10.0;
	
	public static final int MODE_AVERAGE			= 0;
	public static final int MODE_ADD_LIGHT			= 1;
	public static final int MODE_ADD_DARK			= 2;
	public static final int MODE_MAX_LIGHT			= 3;
	public static final int MODE_MIN_LIGHT			= 4;
	public static final int MODE_ADD_GATED_LIGHT	= 5;
	public static final int MODE_ADD_GATED_DARK		= 6;

	public static void plotTimeSeries(ArrayList<ContProbDist> series1, 
			ArrayList<ContProbDist> series2, double minY, double maxY, boolean logarithmicScale,
			int pixelsY, int pixelsX, Color background, ArrayList<Color> colorRamp1,
			ArrayList<Color> colorRamp2, boolean logistic, int mixMode, String file,
			String legendFile, int legendSize, boolean compressionLightening)	throws IOException
	{
		int width					= series1.size();
		if (series2.size() != width)
			throw new IllegalArgumentException("Series are not of the same size");
		if (colorRamp1.size() <= 0)
			throw new IllegalArgumentException("Color ramp 1 has no elements");
		if (colorRamp2.size() <= 0)
			throw new IllegalArgumentException("Color ramp 2 has no elements");
		
		int height					= pixelsY;
		int imageType				= BufferedImage.TYPE_4BYTE_ABGR;
		BufferedImage image			= new BufferedImage(width, height, imageType);
		
		Graphics2D graph			= image.createGraphics();
		graph.setPaint(background);
		graph.fillRect(0, 0, image.getWidth(), image.getHeight());		
		
		for (int x = 0; x < width; x++)
		{
			ContProbDist dist1		= series1.get(x);
			ContProbDist dist2		= series2.get(x);
			
			// Obtain density values
			double max1				= 0.0;
			double max2				= 0.0;
			ArrayList<Double> pdfs1	= new ArrayList<>(pixelsY);
			ArrayList<Double> pdfs2	= new ArrayList<>(pixelsY);
			for (int y = 0; y < pixelsY; y++)
			{
				double base			= (double)y/(pixelsY - 1);
				double val			= logarithmicScale
									? Math.pow(10.0, Math.log10(minY) + base*Math.log10(maxY/minY))
									: minY + base*(maxY - minY);
				double pdf1			= dist1.getpdf(val);
				double pdf2			= dist2.getpdf(val);
				max1				= pdf1 > max1 ? pdf1 : max1;
				max2				= pdf2 > max2 ? pdf2 : max2;
				pdfs1.add(pdf1);
				pdfs2.add(pdf2);
			}
			
			// Assign colors
			for (int y = 0; y < pixelsY; y++)
			{
				double pdf1			= pdfs1.get(y);
				double pdf2			= pdfs2.get(y);
				double fraction1	= pdf1/max1;
				double fraction2	= pdf2/max2;
				if (fraction1 >= DRAW_THRESHOLD && fraction2 < DRAW_THRESHOLD)
				{
					Color color		= getColor(colorRamp1, fraction1);
					image.setRGB(x, pixelsY - y - 1, color.getRGB());
				}
				else if (fraction2 >= DRAW_THRESHOLD && fraction1 < DRAW_THRESHOLD)
				{
					Color color		= getColor(colorRamp2, fraction2);
					image.setRGB(x, pixelsY - y - 1, color.getRGB());
				}
				else if (fraction1 >= DRAW_THRESHOLD && fraction2 >= DRAW_THRESHOLD)
				{
					Color color1	= getColor(colorRamp1, fraction1);
					Color color2	= getColor(colorRamp2, fraction2);
					Color mix = mixColors(color1, color2, mixMode, logistic, fraction1, fraction2);
					image.setRGB(x, pixelsY - y - 1, mix.getRGB());
				}
			}
		}
		
		File outputfile			= new File(file);
		if (pixelsX < width)
		{
			BufferedImage compImage		= new BufferedImage(pixelsX, height, imageType);
			graph						= compImage.createGraphics();
			graph.setPaint(background);
			graph.fillRect(0, 0, image.getWidth(), image.getHeight());
			
			double stretch				= (double)width/pixelsX;
			int pixelsToCombine			= (int)Math.ceil(stretch);
			for (int x = 0; x < pixelsX; x++)
			{
				double centroid			= stretch*x;
				int leftPix				= Math.max(0,		(int)(centroid - pixelsToCombine/2));
				int rightPix			= Math.min(width,	(int)(centroid + pixelsToCombine/2));
				for (int y = 0; y < height; y++)
				{
					Color color			= compressionLightening ? Color.BLACK : Color.WHITE;
					for (int p = leftPix; p <= rightPix; p++)
					{
						Color newColor	= new Color(image.getRGB(p, pixelsY - y - 1));
						if (compressionLightening)
							color		= maxLight(color, newColor);
						else
							color		= minLight(color, newColor);
					}
					compImage.setRGB(x, pixelsY - y - 1, color.getRGB());
				}
			}
			ImageIO.write(compImage, "png", outputfile);
		}
		else
			ImageIO.write(image, "png", outputfile);
			
		
		if (legendFile.equals(""))
			return;
		
		// Create legend
		BufferedImage legendIm	= new BufferedImage(legendSize, legendSize, imageType);
		for		(int x = 0; x < legendSize; x++)
		{
			double fraction1	= (double)x/(legendSize - 1);
			for	(int y = 0; y < legendSize; y++)
			{
				double fraction2 = (double)y/(legendSize - 1);
				Color color1	= getColor(colorRamp1, fraction1);
				Color color2	= getColor(colorRamp2, fraction2);
				Color mix = mixColors(color1, color2, mixMode, logistic, fraction1, fraction2);
				legendIm.setRGB(x, legendSize - y - 1, mix.getRGB());
			}
		}
		outputfile				= new File(legendFile);
		ImageIO.write(legendIm, "png", outputfile);
	}

	private static Color mixColors(Color color1, Color color2, int mixMode, boolean logistic,
									double fraction1, double fraction2)
	{
		Color mix		= null;
		if (mixMode == MODE_AVERAGE)
		{
			double mixFrac = fraction2/(fraction1 + fraction2);
			if (logistic)
				mixFrac	= 1/(1 + Math.exp(-LOGISTIC_STEEPNESS*(mixFrac - 0.5)));
			mix			= average(color1, color2, mixFrac);
		}
		else if (mixMode == MODE_ADD_LIGHT)
			mix			= addLight(color1, color2);
		else if (mixMode == MODE_ADD_DARK)
			mix			= addDarkness(color1, color2);
		else if (mixMode == MODE_MAX_LIGHT)
			mix			= maxLight(color1, color2);
		else if (mixMode == MODE_MIN_LIGHT)
			mix			= minLight(color1, color2);
		else if (mixMode == MODE_ADD_GATED_LIGHT)
			mix			= addGatedLight(color1, color2, fraction1, fraction2);
		else if (mixMode == MODE_ADD_GATED_DARK)
			mix			= addGatedDarkness(color1, color2, fraction1, fraction2);
		return mix;
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
		return average(colorRamp.get(color1), colorRamp.get(color2), innerFraction);
	}
	
	private static Color average(Color color1, Color color2, double fraction)
	{
		fraction		= fraction > 1.0 ? 1.0 : fraction;
		fraction		= fraction < 0.0 ? 0.0 : fraction;
		if (fraction == 0.0)
			return color1;
		if (fraction == 1.0)
			return color2;
		
		int red			= (int)(color1.getRed()*	(1 - fraction) + color2.getRed()*	fraction);
		int green		= (int)(color1.getGreen()*	(1 - fraction) + color2.getGreen()*	fraction);
		int blue		= (int)(color1.getBlue()*	(1 - fraction) + color2.getBlue()*	fraction);
		int alpha		= (int)(color1.getAlpha()*	(1 - fraction) + color2.getAlpha()*	fraction);
		return new Color(red, green, blue, alpha);
	}
	
	private static Color addLight(Color color1, Color color2)
	{
		int red			= Math.min(255,	color1.getRed()		+	color2.getRed()		);
		int green		= Math.min(255,	color1.getGreen()	+	color2.getGreen()	);
		int blue		= Math.min(255,	color1.getBlue()	+	color2.getBlue()	);
		int alpha		= Math.max(		color1.getAlpha(),		color2.getAlpha()	);
		return new Color(red, green, blue, alpha);
	}
	
	private static Color addDarkness(Color color1, Color color2)
	{
		int red		= Math.max(0, 255 - ((510 - color1.getRed()		-	color2.getRed()		)));
		int green	= Math.max(0, 255 - ((510 - color1.getGreen()	-	color2.getGreen()	)));
		int blue	= Math.max(0, 255 - ((510 - color1.getBlue()	-	color2.getBlue()	)));
		int alpha	= Math.max(					color1.getAlpha(),		color2.getAlpha()	);
		return new Color(red, green, blue, alpha);
	}
	
	private static Color maxLight(Color color1, Color color2)
	{
		int red		= Math.max(color1.getRed(),		color2.getRed()		);
		int green	= Math.max(color1.getGreen(),	color2.getGreen()	);
		int blue	= Math.max(color1.getBlue(),	color2.getBlue()	);
		int alpha	= Math.max(color1.getAlpha(),	color2.getAlpha()	);
		return new Color(red, green, blue, alpha);
	}
	
	private static Color minLight(Color color1, Color color2)
	{
		int red		= Math.min(color1.getRed(),		color2.getRed()		);
		int green	= Math.min(color1.getGreen(),	color2.getGreen()	);
		int blue	= Math.min(color1.getBlue(),	color2.getBlue()	);
		int alpha	= Math.max(color1.getAlpha(),	color2.getAlpha()	);
		return new Color(red, green, blue, alpha);
	}
	
	private static Color addGatedLight(Color color1, Color color2,
										double weight1, double weight2)
	{
		Color base, extra;
		double baseW, extraW;
		if (weight1 >= weight2)
		{
			base			= color1;
			extra			= color2;
			baseW			= weight1;
			extraW			= weight2;
		}
		else
		{
			base			= color2;
			extra			= color1;
			baseW			= weight2;
			extraW			= weight1;
		}
		
		int red				= base.getRed();
		int green			= base.getGreen();
		int blue			= base.getBlue();
		int alpha			= Math.max(color1.getAlpha(), color2.getAlpha());
		
		double plusRed		= extra.getRed()	- base.getRed();
		double plusGreen	= extra.getGreen()	- base.getGreen();
		double plusBlue		= extra.getBlue()	- base.getBlue();
		
		double ratio		= extraW/baseW;
		if (plusRed		> 0.0)
			red				+= Math.min(255 - red,		plusRed		)*ratio;
		if (plusGreen	> 0.0)
			green			+= Math.min(255 - green,	plusGreen	)*ratio;
		if (plusBlue	> 0.0)
			blue			+= Math.min(255 - blue, 	plusBlue	)*ratio;
		
		return new Color(red, green, blue, alpha);
	}
	
	private static Color addGatedDarkness(Color color1, Color color2,
											double weight1, double weight2)
	{
		Color base, extra;
		double baseW, extraW;
		if (weight1 >= weight2)
		{
			base			= color1;
			extra			= color2;
			baseW			= weight1;
			extraW			= weight2;
		}
		else
		{
			base			= color2;
			extra			= color1;
			baseW			= weight2;
			extraW			= weight1;
		}
		
		int red				= base.getRed();
		int green			= base.getGreen();
		int blue			= base.getBlue();
		int alpha			= Math.max(color1.getAlpha(), color2.getAlpha());
		
		double minusRed		= base.getRed()		- extra.getRed();
		double minusGreen	= base.getGreen()	- extra.getGreen();
		double minusBlue	= base.getBlue()	- extra.getBlue();
		
		double ratio		= extraW/baseW;
		if (minusRed		> 0.0)
			red				-= Math.min(red,	minusRed	)*ratio;
		if (minusGreen	> 0.0)
			green			-= Math.min(green,	minusGreen	)*ratio;
		if (minusBlue	> 0.0)
			blue			-= Math.min(blue, 	minusBlue	)*ratio;
		
		return new Color(red, green, blue, alpha);
	}

}
