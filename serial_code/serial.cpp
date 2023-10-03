// Libraries for file reading
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>

// CPU timing library
#include <chrono>

#include "rasterization.h"
#include "containers.h"

#define MAPPED_CSV "../T1NA_fixed.csv"
#define OUTPUT_CSV "../rasterized_serial.csv"

void loadPolygonsFromCSV(
	int startLine, int endLine, std::vector<Polygon> &polygons)
{
	std::ifstream fin;
	std::string line, token, coordToken;
	std::vector<Point> points;
	int polyID;
	double x, y;

	printf("Loading dataset '%s'...\n", MAPPED_CSV);
	fin.open(MAPPED_CSV);

	if (!fin.good())
	{
		printf("ERROR: dataset file could not be opened.\n");
		return;
	}

	// Skip lines until the desired starting line.
	for (int i = 1; i < startLine; i++)
	{
		getline(fin, line);
	}

	// Read & parse lines with polygons.
	for (int i = startLine; i <= endLine; i++)
	{
		points.clear();

		// Get the whole line in a stream.
		getline(fin, line);
		std::stringstream lineStream(line);

		// Parse polygon ID at beginning of the line.
		getline(lineStream, token, ',');
		polyID = std::stoi(token);

		// Create new polygon and add it to list.
		polygons.push_back(Polygon());
		polygons[i - 1].recID = polyID;

		// Parse polygon points.
		while (getline(lineStream, token, ','))
		{
			std::stringstream tokenStream(token);

			getline(tokenStream, coordToken, ' ');
			x = std::stod(coordToken);

			getline(tokenStream, coordToken, ' ');
			y = std::stod(coordToken);

			polygons[i - 1].addPoint(x, y);
		}
	}
	printf("Dataset loaded!\n\n");
}

void writeResultsToCSV(std::vector<Polygon> &polygons)
{
	struct hilbertID
	{
		ID id;
		int value;
	};

	std::ofstream fout;
	std::vector<hilbertID> hilbertIDs;
	int pointValue, hilbertX, hilbertY;

	printf("Writing rasterization results to '%s'...\n", OUTPUT_CSV);
	fout.open(OUTPUT_CSV, std::ios::trunc);
	if (fout.is_open())
	{
		// Add header to file.
		fout << "Polygon ID,[Hilbert ID] [1: Partial / 2: Full]\n";

		for (int i = 0; i < polygons.size(); i++)
		{
			for (auto &id : polygons[i].partialCellPackage.hilbertCellIDs)
			{
				hilbertIDs.push_back({id, 1});
			}
			for (auto &id : polygons[i].fullCellPackage.hilbertCellIDs)
			{
				hilbertIDs.push_back({id, 2});
			}

			// Sort by Hilbert ID in ascending order.
			std::sort(
				hilbertIDs.begin(),
				hilbertIDs.end(),
				[](hilbertID a, hilbertID b) // sorting lambda function
				{
					return a.id < b.id;
				});

			// Write polygon info to file.
			fout << polygons[i].recID;
			for (int j = 0; j < hilbertIDs.size(); j++)
			{
				fout << "," << hilbertIDs[j].id << " " << hilbertIDs[j].value;
			}
			fout << "\n";

			hilbertIDs.clear();
		}
		fout.close();
		printf("Writing completed!\n");
	}
	else
	{
		printf("ERROR: output file could not be opened.\n");
	}
}

void calculateMBR(Polygon &poly)
{
	poly.mbr.pMax = Point(poly.vertices[0].x, poly.vertices[0].y);
	poly.mbr.pMin = Point(poly.vertices[0].x, poly.vertices[0].y);

	for (auto &p : poly.vertices)
	{
		if (p.x < poly.mbr.pMin.x)
		{
			poly.mbr.pMin.x = p.x;
		}
		if (p.y < poly.mbr.pMin.y)
		{
			poly.mbr.pMin.y = p.y;
		}
		if (p.x > poly.mbr.pMax.x)
		{
			poly.mbr.pMax.x = p.x;
		}
		if (p.y > poly.mbr.pMax.y)
		{
			poly.mbr.pMax.y = p.y;
		}
	}

	// Round MBR and add 1 cell buffer around
	poly.mbr.pMin.x = (int)poly.mbr.pMin.x - 1;
	poly.mbr.pMin.y = (int)poly.mbr.pMin.y - 1;
	poly.mbr.pMax.x = (int)poly.mbr.pMax.x + 1;
	poly.mbr.pMax.y = (int)poly.mbr.pMax.y + 1;
}

int main()
{
	std::vector<Polygon> polygons;
	Section sec;
	std::chrono::milliseconds mbrTotal, rasterTotal, duration;
	std::chrono::_V2::system_clock::time_point start, stop, pStart, pStop;

	mbrTotal = rasterTotal = std::chrono::milliseconds::zero();

	int startLine = 1;	// Start from 1
	int endLine = 1000; // Max line: 123045

	sec.rasterxMin = -124.849;
	sec.rasteryMin = 24.5214;
	sec.rasterxMax = -66.8854;
	sec.rasteryMax = 49.3844;

	loadPolygonsFromCSV(startLine, endLine, polygons);

	start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i <= endLine - startLine; i++)
	{
		printf("\rRasterizing polygon %d of %d (ID: %d)",
			   i + 1, endLine - startLine + 1, polygons[i].recID);

		pStart = std::chrono::high_resolution_clock::now();
		calculateMBR(polygons[i]);
		pStop = std::chrono::high_resolution_clock::now();
		mbrTotal += std::chrono::duration_cast<std::chrono::milliseconds>(pStop - pStart);

		pStart = std::chrono::high_resolution_clock::now();
		rasterizeSimple(polygons[i], sec);
		pStop = std::chrono::high_resolution_clock::now();
		rasterTotal += std::chrono::duration_cast<std::chrono::milliseconds>(pStop - pStart);
	}
	stop = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	std::cout << "\n\nTotal dataset time: " << duration.count() << " ms" << std::endl;
	std::cout << "Total MBR calculation time: " << mbrTotal.count() << " ms" << std::endl;
	std::cout << "Total rasterization time: " << rasterTotal.count() << " ms" << std::endl;

	writeResultsToCSV(polygons);
}