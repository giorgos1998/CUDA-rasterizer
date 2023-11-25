#include "rasterization.h"

/* GLOBALS */
vector<pair<uint,uint>> lowGranularityFullCellsIndices;

/*
*-------------------------------------------------------
*
*     UTILITY
*       
*
*-------------------------------------------------------
*/


void printMatrix(uint **M, uint &WIDTH, uint &HEIGHT){
	cout << "Dimensions (" << WIDTH << " x " << HEIGHT << "): " << endl;
	for(int j=HEIGHT-1; j>=0; j--){
		for(int i=0; i<WIDTH; i++){
			cout << M[i][j] << " ";
		}
		cout << endl;
	}
}

double normalizeSingleValueToSectionHilbert(double val, double minval, double maxval, uint &orderN){
	double newval =  ((double) (orderN-1) / (maxval - minval)) * (val - minval);
	if(newval < 0){
		newval = 0;
	}
	if(newval >= orderN){
		newval = orderN-1;
	}
	return newval;
}	

//N = 12,13...16
void normalizeXYOrderPower(double &x, double &y, uint &orderN, uint &newOrderN){
	double amount = pow(2, (orderN - newOrderN));
	if(orderN < newOrderN){
		x = x * amount;
		y = y * amount;
	}else{
		x = x / amount;
		y = y / amount;
	}
}

//N = 12,13...16
void normalizeXYOrderPower(uint &x, uint &y, uint &orderN, uint &newOrderN){
	double amount = pow(2, (orderN - newOrderN));
	if(orderN < newOrderN){
		x = x * amount;
		y = y * amount;
	}else{
		x = x / amount;
		y = y / amount;
	}
}

void normalizeXYToSectionHilbert(double &x, double &y, double minX, double minY, double maxX, double maxY, uint &orderN){
	x = ((double) (orderN-1) / (maxX - minX)) * (x - minX);
	y = ((double) (orderN-1) / (maxY - minY)) * (y - minY);

	if(x < 0){
		x = 0;
	}
	if(y < 0){
		y = 0;
	}

	if(x >= orderN){
		x = orderN-1;
	}
	if(y >= orderN){
		y = orderN-1;
	}
}

void mapPolygonCustom(Polygon &pol, uint orderN, Section &sec){
	//first, map the polygon's coordinates to this section's hilbert space
	for(auto &p: pol.vertices){
		normalizeXYToSectionHilbert(p.x, p.y, sec.rasterxMin, sec.rasteryMin, sec.rasterxMax, sec.rasteryMax, orderN);
		pol.boostPolygon.outer().emplace_back(p.x, p.y);
	}
	boost::geometry::correct(pol.boostPolygon);	
	//map the mbr
	pol.minHilbertX = normalizeSingleValueToSectionHilbert(pol.mbr.pMin.x, sec.rasterxMin, sec.rasterxMax, orderN);
	pol.minHilbertX > 0 ? pol.minHilbertX -= 1 : pol.minHilbertX = 0;
	pol.minHilbertY = normalizeSingleValueToSectionHilbert(pol.mbr.pMin.y, sec.rasteryMin, sec.rasteryMax, orderN);
	pol.minHilbertY > 0 ? pol.minHilbertY -= 1 : pol.minHilbertY = 0;
	pol.maxHilbertX = normalizeSingleValueToSectionHilbert(pol.mbr.pMax.x, sec.rasterxMin, sec.rasterxMax, orderN);
	pol.maxHilbertX < orderN - 1 ? pol.maxHilbertX += 1 : pol.maxHilbertX = orderN - 1;
	pol.maxHilbertY = normalizeSingleValueToSectionHilbert(pol.mbr.pMax.y, sec.rasteryMin, sec.rasteryMax, orderN);
	pol.maxHilbertY < orderN - 1 ? pol.maxHilbertY += 1 : pol.maxHilbertY = orderN - 1;

	// //set dimensions for buffers (matrices)
	pol.bufferWidth = pol.maxHilbertX - pol.minHilbertX + 1;
	pol.bufferHeight = pol.maxHilbertY - pol.minHilbertY + 1;
}

/* WARNING: fromN and toN are 2^N, not N */
void mapPolygonHilbertToHilbertWithControl(Polygon &pol, uint fromN, uint toN, uint powerFrom, uint &powerTo){
	ID d;
	pol.boostPolygon.outer().clear();
	//first, map the polygon's coordinates to this section's hilbert space
	for(auto &p: pol.vertices){
		normalizeXYOrderPower(p.x, p.y, powerFrom, powerTo);		
		pol.boostPolygon.outer().emplace_back(p.x, p.y);

	}
	boost::geometry::correct(pol.boostPolygon);	

	normalizeXYOrderPower(pol.minHilbertX, pol.minHilbertY, powerFrom, powerTo);
	pol.minHilbertX > 0 ? pol.minHilbertX -= 1 : pol.minHilbertX = 0;
	pol.minHilbertY > 0 ? pol.minHilbertY -= 1 : pol.minHilbertY = 0;
	normalizeXYOrderPower(pol.maxHilbertX, pol.maxHilbertY, powerFrom, powerTo);
	pol.maxHilbertX < toN - 1 ? pol.maxHilbertX += 1 : pol.maxHilbertX = toN - 1;
	pol.maxHilbertY < toN - 1 ? pol.maxHilbertY += 1 : pol.maxHilbertY = toN - 1;

	// // //set dimensions for buffers (matrices)
	pol.bufferWidth = pol.maxHilbertX - pol.minHilbertX + 1;
	pol.bufferHeight = pol.maxHilbertY - pol.minHilbertY + 1;
}


void push(std::vector<ID>& stack, ID x, ID y)
{
  // C++'s std::vector can act as a stack and manage memory for us
  stack.push_back(x);
  stack.push_back(y);
}

bool pop(std::vector<ID>& stack, ID& x, ID& y)
{
  if(stack.size() < 2) return false; // it's empty
  y = stack.back();
  stack.pop_back();
  x = stack.back();
  stack.pop_back();
  return true;
}

/*
*
*
*	Amanatides and Wu algorithm (grid traversal - modified DDA)
*
*
*/

inline bool checkY(double &y1, double &OGy1, double &OGy2, uint &startCellY, uint &endCellY){
	if(OGy1 < OGy2){
		return (startCellY <= y1 && y1 <= endCellY + 1);
	}else{
		return (endCellY <= y1 && y1 <= startCellY + 1);
	}
}

uint** calculatePartialCellsCUSTOM(Polygon &pol, uint &orderN){
	//create empty matrix
	uint **M = new uint*[pol.bufferWidth]();
	for(uint i=0; i<pol.bufferWidth; i++){
		M[i] = new uint[pol.bufferHeight]();		
		for(uint j=0; j<pol.bufferHeight; j++){
			M[i][j] = EMPTY_COLOR;
		}
	}

	//local variables
	uint startCellX, startCellY, endCellX, endCellY;;
	uint currentCellX, currentCellY;
	double x1, y1, x2, y2;
	double ogy1, ogy2;
	int stepX, stepY;
	uint verticalStartY, verticalEndY, horizontalY;
	double tMaxX, tMaxY;
	double tDeltaX, tDeltaY;
	double edgeLength;
	deque<point_xy> output;
	double error_margin = 0.00001;

	//loop points
	for (auto it = pol.vertices.begin(); it != pol.vertices.end()-1; it++) {
		//set points x1 has the lowest x
		if(it->x < (it+1)->x){
			x1 = it->x;
			y1 = it->y;
			x2 = (it+1)->x;
			y2 = (it+1)->y;
		}else{
			x2 = it->x;
			y2 = it->y;
			x1 = (it+1)->x;
			y1 = (it+1)->y;
		}

		//keep original values of y1,y2
		ogy1 = y1;
		ogy2 = y2;	

		//set endpoint hilbert cells
		startCellX = (int)x1;
		startCellY = (int)y1;
		endCellX = (int)x2;
		endCellY = (int)y2;

		if(startCellX == endCellX && startCellY == endCellY){
			//label the cell in the partially covered matrix
			M[startCellX-pol.minHilbertX][startCellY-pol.minHilbertY] = PARTIAL_COLOR;

		}else{
			//set step (based on direction)
			stepX = x2 > x1 ? 1 : -1;
			stepY = y2 > y1 ? 1 : -1;

			//define the polygon edge
			linestring ls{{x1, y1},{x2, y2}};
			edgeLength = boost::geometry::length(ls);

			//define NEAREST VERTICAL grid line
			linestring vertical{{startCellX+1, 0},{startCellX+1, orderN}};
			
			//define NEAREST HORIZONTAL grid line
			y1 < y2 ? horizontalY = int(y1) + 1 : horizontalY = int(y1);
			linestring horizontal{{0, horizontalY},{orderN, horizontalY}};

			// cout << "vertical line: (" << boost::geometry::get<0,0>(vertical) << "," << boost::geometry::get<0,1>(vertical) << "),(" << boost::geometry::get<1,0>(vertical) << "," << boost::geometry::get<1,1>(vertical) << ")" << endl;
			// cout << "horizontal line: (" << boost::geometry::get<0,0>(horizontal) << "," << boost::geometry::get<0,1>(horizontal) << "),(" << boost::geometry::get<1,0>(horizontal) << "," << boost::geometry::get<1,1>(horizontal) << ")" << endl;

			//get intersection points with the vertical and nearest lines
			boost::geometry::intersection(ls, vertical, output);
			point_xy intersectionPointVertical = output[0];
			output.clear();
			boost::geometry::intersection(ls, horizontal, output);
			point_xy intersectionPointHorizontal = output[0];
			output.clear();

			// //keep in mind: the line segment may not intersect a vertical or horizontal line!!!!!!!!
			// cout << "vertical intersection point distance from line segment: " << boost::geometry::distance(intersectionPointVertical, ls) << endl;
			// cout << "horizontal intersection point distance from line segment: " << boost::geometry::distance(intersectionPointHorizontal, ls) << endl;
			//tmax
			if(boost::geometry::distance(intersectionPointVertical, ls) <= error_margin && boost::geometry::distance(intersectionPointVertical, vertical) <= error_margin){
				linestring tXMaxline{{x1,y1},intersectionPointVertical};
				tMaxX = boost::geometry::length(tXMaxline);
			}else{
				// cout << fixed << setprecision(10) << "tMaxX edgeLength: " << edgeLength << endl;
				tMaxX = edgeLength;
			}
			if(boost::geometry::distance(intersectionPointHorizontal, ls) <= error_margin && boost::geometry::distance(intersectionPointHorizontal, horizontal) <= error_margin){
				linestring tYMaxline{{x1,y1},intersectionPointHorizontal};
				tMaxY = boost::geometry::length(tYMaxline);
			}else{
				tMaxY = edgeLength;
			}

			//deltas
			tDeltaX = edgeLength / (x2 - x1);
			tDeltaY = edgeLength / abs(y2 - y1);

			//loop (traverse ray)
			while(startCellX <= x1 && x1 < endCellX+1 && checkY(y1, ogy1, ogy2, startCellY, endCellY)){				
				M[(int)x1-pol.minHilbertX][(int)y1-pol.minHilbertY] = PARTIAL_COLOR;				
				if(tMaxX < tMaxY){
					x1 = x1 + stepX;
					tMaxX = tMaxX + tDeltaX;
				}else{
					y1 = y1 + stepY;
					tMaxY = tMaxY + tDeltaY;
				}				
			}
		}
	}

	return M;
}

/*

	FLOOD FILL


*/

void floodFillUncertainSpecific(uint **M, ID x, ID y, uint &WIDTH, uint &HEIGHT, Polygon &pol, uint NEW_COLOR){
	int x1;
	bool spanAbove, spanBelow;
	ID d;
	std::vector<ID> stack;
	push(stack, x, y);

	while(pop(stack, x, y)){
		x1 = x;
		//go to the start of the line, where to begine painting
		while(x1 >= 0 && (M[x1][y] == UNCERTAIN_COLOR)){
			x1--;
		}
		x1++;
		spanAbove = spanBelow = 0;

		//paint rows (full)
		while(x1 < WIDTH && M[x1][y] == UNCERTAIN_COLOR){
			//this line paints
			M[x1][y] = NEW_COLOR;

			//store full to full package
			if(NEW_COLOR == FULL_COLOR){
				d = xy2d(HILBERT_n, x1+pol.minHilbertX, y+pol.minHilbertY);
				pol.fullCellPackage.addID(d);
			}

			if(!spanAbove && y > 0 && M[x1][y - 1] == UNCERTAIN_COLOR){
				push(stack, x1, y - 1);
				spanAbove = 1;
			}else if(spanAbove && y > 0 && M[x1][y - 1] != UNCERTAIN_COLOR){
				spanAbove = 0;
			}

			if(!spanBelow && y < HEIGHT - 1 && M[x1][y + 1] == UNCERTAIN_COLOR){
				push(stack, x1, y + 1);
				spanBelow = 1;
			}else if(spanBelow && y < HEIGHT - 1 && M[x1][y + 1] != UNCERTAIN_COLOR){
				spanBelow = 0;
			}
			x1++;
		}
	}
}

uint** calculatePartialAndUncertain(Polygon &pol, uint &orderN){
	//create empty matrix
	uint **M = new uint*[pol.bufferWidth]();
	for(uint i=0; i<pol.bufferWidth; i++){
		M[i] = new uint[pol.bufferHeight]();		
		for(uint j=0; j<pol.bufferHeight; j++){
			M[i][j] = UNCERTAIN_COLOR;
		}
	}

	// cout << pol.bufferWidth << "x" << pol.bufferHeight << endl;

	//local variables
	uint startCellX, startCellY, endCellX, endCellY;;
	uint currentCellX, currentCellY;
	double x1, y1, x2, y2;
	double ogy1, ogy2;
	int stepX, stepY;
	uint verticalStartY, verticalEndY, horizontalY;
	double tMaxX, tMaxY;
	double tDeltaX, tDeltaY;
	double edgeLength;
	deque<point_xy> output;
	double error_margin = 0.00001;

	//loop points
	for (auto it = pol.vertices.begin(); it != pol.vertices.end()-1; it++) {
		//set points x1 has the lowest x
		if(it->x < (it+1)->x){
			x1 = it->x;
			y1 = it->y;
			x2 = (it+1)->x;
			y2 = (it+1)->y;
		}else{
			x2 = it->x;
			y2 = it->y;
			x1 = (it+1)->x;
			y1 = (it+1)->y;
		}

		//keep original values of y1,y2
		ogy1 = y1;
		ogy2 = y2;	

		//set endpoint hilbert cells
		startCellX = (int)x1;
		startCellY = (int)y1;
		endCellX = (int)x2;
		endCellY = (int)y2;

		// if(startCellX == 34672){
		// 	cout << fixed << setprecision(10) << "Edge: (" << x1 << "," << y1 << "),(" << x2 << "," << y2 << ")" << endl;
		// }
		// cout << fixed << setprecision(10) << "Edge: (" << x1 << "," << y1 << "),(" << x2 << "," << y2 << ")" << endl;
		
		// cout << "  StartCell: (" << startCellX << "," << startCellY << ")" << endl;
		// cout << "  EndCell: (" << endCellX << "," << endCellY << ")" << endl;

		if(startCellX == endCellX && startCellY == endCellY){
			//label the cell in the partially covered matrix
			M[startCellX-pol.minHilbertX][startCellY-pol.minHilbertY] = PARTIAL_COLOR;
		}else{
			//set step (based on direction)
			stepX = x2 > x1 ? 1 : -1;
			stepY = y2 > y1 ? 1 : -1;

			//define the polygon edge
			linestring ls{{x1, y1},{x2, y2}};
			edgeLength = boost::geometry::length(ls);

			//define NEAREST VERTICAL grid line
			linestring vertical{{startCellX+1, 0},{startCellX+1, orderN}};
			
			//define NEAREST HORIZONTAL grid line
			y1 < y2 ? horizontalY = int(y1) + 1 : horizontalY = int(y1);
			linestring horizontal{{0, horizontalY},{orderN, horizontalY}};

			// cout << "vertical line: (" << boost::geometry::get<0,0>(vertical) << "," << boost::geometry::get<0,1>(vertical) << "),(" << boost::geometry::get<1,0>(vertical) << "," << boost::geometry::get<1,1>(vertical) << ")" << endl;
			// cout << "horizontal line: (" << boost::geometry::get<0,0>(horizontal) << "," << boost::geometry::get<0,1>(horizontal) << "),(" << boost::geometry::get<1,0>(horizontal) << "," << boost::geometry::get<1,1>(horizontal) << ")" << endl;

			//get intersection points with the vertical and nearest lines
			boost::geometry::intersection(ls, vertical, output);
			point_xy intersectionPointVertical = output[0];
			output.clear();
			boost::geometry::intersection(ls, horizontal, output);
			point_xy intersectionPointHorizontal = output[0];
			output.clear();

			// //keep in mind: the line segment may not intersect a vertical or horizontal line!!!!!!!!
			// cout << "vertical intersection point distance from line segment: " << boost::geometry::distance(intersectionPointVertical, ls) << endl;
			// cout << "horizontal intersection point distance from line segment: " << boost::geometry::distance(intersectionPointHorizontal, ls) << endl;
			//tmax
			if(boost::geometry::distance(intersectionPointVertical, ls) <= error_margin && boost::geometry::distance(intersectionPointVertical, vertical) <= error_margin){
				linestring tXMaxline{{x1,y1},intersectionPointVertical};
				tMaxX = boost::geometry::length(tXMaxline);
			}else{
				// cout << fixed << setprecision(10) << "tMaxX edgeLength: " << edgeLength << endl;
				tMaxX = edgeLength;
			}
			if(boost::geometry::distance(intersectionPointHorizontal, ls) <= error_margin && boost::geometry::distance(intersectionPointHorizontal, horizontal) <= error_margin){
				linestring tYMaxline{{x1,y1},intersectionPointHorizontal};
				tMaxY = boost::geometry::length(tYMaxline);
			}else{
				tMaxY = edgeLength;
				// cout << fixed << setprecision(10) << "tMaxY edgeLength: " << edgeLength << endl;
				// linestring tYMaxline{{x1,y1},intersectionPointHorizontal};
				// cout << fixed << setprecision(10) << "  otherwise it would be " << boost::geometry::length(tYMaxline) << endl;
			}

			//deltas
			tDeltaX = edgeLength / (x2 - x1);
			tDeltaY = edgeLength / abs(y2 - y1);			
			// cout << "deltas: " << tDeltaX << "," << tDeltaY << endl;	
			
			//loop (traverse ray)
			while(startCellX <= x1 && x1 < endCellX+1 && checkY(y1, ogy1, ogy2, startCellY, endCellY)){

				// cout << fixed << setprecision(0) << "(" <<(int)x1 << "," << (int)y1 << ")" << endl;	
				
				M[(int)x1-pol.minHilbertX][(int)y1-pol.minHilbertY] = PARTIAL_COLOR;

				// cout << fixed << setprecision(10) << "  tMaxX: " << tMaxX << endl;
				// cout << fixed << setprecision(10) << "  tMaxY: " << tMaxY << endl;
				
				if(tMaxX < tMaxY){
					x1 = x1 + stepX;
					tMaxX = tMaxX + tDeltaX;
					// cout << " increasing X" << endl;
				}else{
					y1 = y1 + stepY;
					tMaxY = tMaxY + tDeltaY;
					// cout << " increasing Y" << endl;
				}
				// cout << fixed << setprecision(0) << "next voxel(" << (int)x1 << "," << (int)y1 << ")" << endl;
				
			}
		}
	}
	return M;
}


void fillAndFinalizeMatrix(Polygon &pol, uint **M){
	for(int i=0; i<pol.bufferWidth; i++){
		for(int j=0; j<pol.bufferHeight; j++){
			if(M[i][j] == UNCERTAIN_COLOR){
				point_xy p(i+pol.minHilbertX, j+pol.minHilbertY);
				if(boost::geometry::within(p, pol.boostPolygon)){
					//flood fill FULL
					floodFillUncertainSpecific(M, i, j, pol.bufferWidth, pol.bufferHeight, pol, FULL_COLOR);
				}else{
					//flood fill EMPTY
					floodFillUncertainSpecific(M, i, j, pol.bufferWidth, pol.bufferHeight, pol, EMPTY_COLOR);
				}
			}else if(M[i][j] == PARTIAL_COLOR){			
				//store into preallocated array
				pol.partialCellPackage.addID(xy2d(HILBERT_n, i + pol.minHilbertX, j + pol.minHilbertY));
			}
		}

	}
}

void finalizePartialsFromMatrix(Polygon &pol, uint **M){
	for(int i=0; i<pol.bufferWidth; i++){
		for(int j=0; j<pol.bufferHeight; j++){			
			if(M[i][j] == PARTIAL_COLOR){			
				//store into preallocated array
				pol.uncompressedALL.push_back(xy2d(HILBERT_n, i + pol.minHilbertX, j + pol.minHilbertY));
			}
		}

	}

	//sort the cell IDs 
	sort(pol.uncompressedALL.begin(), pol.uncompressedALL.end());
	pol.numIntervalsALL = pol.uncompressedALL.size();
	pol.numBytesALL = pol.uncompressedALL.size();
	pol.numIntervalsF = 0;
}

/* 
*
*
*
*	----- 2-GRID RASTERIZATION -----
*
*
*
*/

uint** reduceMatrix(uint **M, uint &fromW, uint &fromH, uint &toW, uint &toH, uint &orderDifference){
	uint indexI, indexJ;	
	uint **newM = new uint*[toW]();
	for(uint i=0; i<toW; i++){
		newM[i] = new uint[toH]();		
		for(uint j=0; j<toH; j++){
			newM[i][j] = EMPTY_COLOR;
		}
	}
	for(uint i=0; i<fromW; i++){
		for(uint j=0; j<fromH; j++){			
			if(M[i][j] == PARTIAL_COLOR){
				indexI = i >> orderDifference;
				indexJ = j >> orderDifference;
				newM[indexI][indexJ] = PARTIAL_COLOR;
			}
		}
	}

	return newM;
}

void transferFullCheckedElements(uint **fromM, uint &fromW, uint &fromH, uint **toM, uint &toW, uint &toH, uint &diffOrder){
	uint indexI, indexJ;
	for(uint i=0; i<fromW; i++){
		for(uint j=0; j<fromH; j++){		
			if(fromM[i][j] == FULL_CHECKED){
				indexI = i << diffOrder;
				indexJ = j << diffOrder;

				for(int ii=indexI; ii<indexI+pow(2,diffOrder); ii++){
					for(int jj=indexJ; jj<indexJ+pow(2,diffOrder); jj++){
						toM[ii][jj] = FULL_CHECKED;
					}
				}
			}else if(fromM[i][j] == PARTIAL_COLOR){
				indexI = i << diffOrder;
				indexJ = j << diffOrder;	

				for(int ii=indexI; ii<indexI+pow(2,diffOrder); ii++){
					for(int jj=indexJ; jj<indexJ+pow(2,diffOrder); jj++){
						if(toM[ii][jj] != PARTIAL_COLOR){
							toM[ii][jj] = UNCERTAIN_COLOR;
						}

					}
				}


			}
		}
	}
}

void floodFillUncertainFULLCHECKED(uint **M, ID x, ID y, uint &WIDTH, uint &HEIGHT, Polygon &pol, uint NEW_COLOR){
	int x1;
	bool spanAbove, spanBelow;
	ID d;
	std::vector<ID> stack;
	push(stack, x, y);

	while(pop(stack, x, y)){
		x1 = x;
		//go to the start of the line, where to begine painting
		while(x1 >= 0 && (M[x1][y] == EMPTY_COLOR)){
			x1--;
		}
		x1++;
		spanAbove = spanBelow = 0;

		//paint rows (full)
		while(x1 < WIDTH && M[x1][y] == EMPTY_COLOR){
			//this line paints
			M[x1][y] = NEW_COLOR;

			//store full to full package
			if(NEW_COLOR == FULL_CHECKED){
				lowGranularityFullCellsIndices.emplace_back(x1,y);
			}

			if(!spanAbove && y > 0 && M[x1][y - 1] == EMPTY_COLOR){
				push(stack, x1, y - 1);
				spanAbove = 1;
			}else if(spanAbove && y > 0 && M[x1][y - 1] != EMPTY_COLOR){
				spanAbove = 0;
			}

			if(!spanBelow && y < HEIGHT - 1 && M[x1][y + 1] == EMPTY_COLOR){
				push(stack, x1, y + 1);
				spanBelow = 1;
			}else if(spanBelow && y < HEIGHT - 1 && M[x1][y + 1] != EMPTY_COLOR){
				spanBelow = 0;
			}
			x1++;
		}
	}
}

void fillAndFinalizeFULLCHECKED(Polygon &pol, uint **M, uint &WIDTH, uint &HEIGHT){
	for(int i=0; i<WIDTH; i++){
		for(int j=0; j<HEIGHT; j++){
			if(M[i][j] == EMPTY_COLOR){
				point_xy p(i+pol.minHilbertX, j+pol.minHilbertY);

				if(boost::geometry::within(p, pol.boostPolygon)){
					//flood fill FULL
					floodFillUncertainFULLCHECKED(M, i, j, WIDTH, HEIGHT, pol, FULL_CHECKED);
				}else{
					//flood fill EMPTY
					floodFillUncertainSpecific(M, i, j, WIDTH, HEIGHT, pol, EMPTY_COLOR);
				}
			}
		}

	}
}

void floodFillClearEmpty(uint **M, ID x, ID y, uint &WIDTH, uint &HEIGHT){
	int x1;
	bool spanAbove, spanBelow;

	std::vector<ID> stack;
	push(stack, x, y);

	ID d;

	while(pop(stack, x, y)){
		x1 = x;
		//go to the start of the line, where to begine painting
		while(x1 >= 0 && (M[x1][y] == UNCERTAIN_COLOR)){
			x1--;
		}
		x1++;
		spanAbove = spanBelow = 0;

		//paint rows (full)
		while(x1 < WIDTH && (M[x1][y] == UNCERTAIN_COLOR)){
			//this line paints
			M[x1][y] = EMPTY_COLOR;

			if(!spanAbove && y > 0 && M[x1][y - 1] == UNCERTAIN_COLOR){
				push(stack, x1, y - 1);
				spanAbove = 1;
			}else if(spanAbove && y > 0 && M[x1][y - 1] != UNCERTAIN_COLOR){
				spanAbove = 0;
			}

			if(!spanBelow && y < HEIGHT - 1 && M[x1][y + 1] == UNCERTAIN_COLOR){
				push(stack, x1, y + 1);
				spanBelow = 1;
			}else if(spanBelow && y < HEIGHT - 1 && M[x1][y + 1] != UNCERTAIN_COLOR){
				spanBelow = 0;
			}
			x1++;
		}
	}
}

void floodFillFinal(uint **M, ID x, ID y, uint &WIDTH, uint &HEIGHT, Polygon &pol, uint &orderN){
	int x1;
	bool spanAbove, spanBelow;

	std::vector<ID> stack;
	push(stack, x, y);

	ID d;

	while(pop(stack, x, y)){
		x1 = x;
		//go to the start of the line, where to begine painting
		while(x1 >= 0 && (M[x1][y] == UNCERTAIN_COLOR || M[x1][y] == EMPTY_COLOR)){
			x1--;
		}
		x1++;
		spanAbove = spanBelow = 0;

		//paint rows (full)
		while(x1 < WIDTH && (M[x1][y] == UNCERTAIN_COLOR || M[x1][y] == EMPTY_COLOR)){
			//this line paints
			M[x1][y] = FULL_COLOR;

			//store (rotation here)
			d = xy2d(orderN, x1+pol.minHilbertX, y+pol.minHilbertY);
			pol.fullCellPackage.hilbertCellIDs.emplace_back(d);

			if(!spanAbove && y > 0 && (M[x1][y - 1] == UNCERTAIN_COLOR || M[x1][y - 1] == EMPTY_COLOR)){
				push(stack, x1, y - 1);
				spanAbove = 1;
			}else if(spanAbove && y > 0 && (M[x1][y - 1] != UNCERTAIN_COLOR || M[x1][y - 1] != EMPTY_COLOR)){
				spanAbove = 0;
			}

			if(!spanBelow && y < HEIGHT - 1 && (M[x1][y + 1] == UNCERTAIN_COLOR || M[x1][y + 1] == EMPTY_COLOR)){
				push(stack, x1, y + 1);
				spanBelow = 1;
			}else if(spanBelow && y < HEIGHT - 1 && (M[x1][y + 1] != UNCERTAIN_COLOR || M[x1][y + 1] != EMPTY_COLOR)){
				spanBelow = 0;
			}
			x1++;
		}
	}
}


void cleanUncertainFinal(uint **M, uint &WIDTH, uint &HEIGHT, Polygon &pol, uint &orderN){
	ID d;
	
	for(int i=0; i<WIDTH; i++){
		for(int j=0; j<HEIGHT; j++){

			if(M[i][j] == UNCERTAIN_COLOR){
				//full color = uncertain at this stage
				
				point_xy p(i+pol.minHilbertX, j+pol.minHilbertY);
				if(boost::geometry::within(p, pol.boostPolygon)){
					//flood fill from this point FULL
					// cout << "Started flood filling FULL from cell " << i << "," << j << endl;
					floodFillFinal(M, i, j, WIDTH, HEIGHT, pol, orderN);
				}
				else{
					//flood fill from this point EMPTY
					floodFillClearEmpty(M, i, j, WIDTH, HEIGHT);
				}
			}else if(M[i][j] == PARTIAL_COLOR){
				pol.partialCellPackage.hilbertCellIDs.emplace_back(xy2d(orderN, i+pol.minHilbertX, j+pol.minHilbertY));
			}
		}

	}
}



void rasterize2grid(Polygon &pol, Section &sec, int &fromPower){
	ID x,y, d;
	clock_t timer;
	
	uint powerOfN = fromPower;
	uint orderN = pow(2,powerOfN);

	//rasterize to the power that was decided
	uint newPowerOfN = pol.orderN;
	uint newOrderN = pow(2,newPowerOfN);

	uint diffOrder = newPowerOfN - powerOfN;
	uint diffOrderPower = pow(2,diffOrder);
	uint bitsToShift = 2 * diffOrder;

	//clear globals
	lowGranularityFullCellsIndices.clear();

	// //map to 2^16
	mapPolygonCustom(pol, HILBERT_n, sec);

	// //HIGH GRANULARITY GRID INFO
	uint highMinX = pol.minHilbertX;
	uint highMinY = pol.minHilbertY;
	uint highMaxX = pol.maxHilbertX;
	uint highMaxY = pol.maxHilbertY;
	uint highGridWidth = highMaxX - highMinX;
	uint highGridHeight = highMaxY - highMinY;

	// LOW GRANULARITY GRID INFO
	uint lowMinX, lowMinY, lowMaxX, lowMaxY;
	uint tempXmin, tempYmin, tempXmax, tempYmax;
	if((highMinX >> diffOrder) > 0){
		lowMinX = (highMinX >> diffOrder) - 1;
	}else{
		lowMinX = 0;
	}

	if((highMinY >> diffOrder) > 0){
		lowMinY = (highMinY >> diffOrder) - 1;
	}else{
		lowMinY = 0;
	}

	if((highMaxX >> diffOrder) < orderN-1){
		lowMaxX = (highMaxX >> diffOrder) + 1;
	}else{
		lowMaxX = orderN-1;
	}

	if((highMaxY >> diffOrder) < orderN-1){
		lowMaxY = (highMaxY >> diffOrder) + 1;
	}else{
		lowMaxY = orderN-1;
	}
	uint lowGridWidth = lowMaxX - lowMinX;
	uint lowGridHeight = lowMaxY - lowMinY;

	tempXmin = lowMinX << diffOrder;
	tempYmin = lowMinY << diffOrder;
	tempXmax = lowMaxX << diffOrder;
	tempYmax = lowMaxY << diffOrder;

	if(tempXmin != highMinX){
		highMinX = tempXmin;
		pol.minHilbertX = highMinX;
	}
	if(tempYmin != highMinY){
		highMinY = tempYmin;
		pol.minHilbertY = highMinY;
	}

	if(tempXmax != highMaxX){
		if(tempXmax > newOrderN-1){
			highMaxX = newOrderN-1;
		}else{
			highMaxX = tempXmax + diffOrderPower;
		}
		pol.maxHilbertX = highMaxX;
	}
	if(tempYmax != highMaxY){
		if(tempYmax > newOrderN-1){
			highMaxY = newOrderN-1;
		}else{
			highMaxY = tempYmax + diffOrderPower;
		}
		pol.maxHilbertY = highMaxY;
	}

	highGridWidth = highMaxX - highMinX;
	highGridHeight = highMaxY - highMinY;
	pol.bufferWidth = highGridWidth;
	pol.bufferHeight = highGridHeight;
	//fix low grid
	lowMinX = highMinX >> diffOrder;
	lowMinY = highMinY >> diffOrder;
	lowMaxX = highMaxX >> diffOrder;
	lowMaxY = highMaxY >> diffOrder;
	lowGridWidth = lowMaxX - lowMinX;
	lowGridHeight = lowMaxY - lowMinY;

	// cout << "FINAL GRIDS: " << endl;
	// cout << "large grid (" << highGridWidth << "x" << highGridHeight << "): (" << highMinX << "," << highMinY << "),(" << highMaxX << "," << highMaxY << ")" << endl;
	// cout << "small grid (" << lowGridWidth << "x" << lowGridHeight << "): (" << lowMinX << "," << lowMinY << "),(" << lowMaxX << "," << lowMaxY << ")" << endl;

	/* PARTIAL CELLS */
	//grid traversal algorithm - create partially covered cell matrix
	uint **M = calculatePartialCellsCUSTOM(pol, newOrderN);
	//print M
	// cout << "Initial matrix (" << pol.bufferWidth << "x" << pol.bufferHeight << "): " << endl;
	// printMatrix(M, pol.bufferWidth, pol.bufferHeight);
	
	//transform to lower granularity
	uint **lowM = reduceMatrix(M, highGridWidth, highGridHeight, lowGridWidth, lowGridHeight, diffOrder);
	// // //print M
	// cout << "Post transformation matrix: " << endl;
	// printMatrix(lowM, lowGridWidth, lowGridHeight);

	//map polygon to 2^12 in order to perform the PiP test for the clean uncertain process
	Polygon mappedPol = pol;
	mapPolygonHilbertToHilbertWithControl(pol, newOrderN, orderN, newPowerOfN, powerOfN);
	// reduceCoordinates(pol, newPowerOfN - powerOfN);

	//fix the boundaries
	pol.minHilbertX = lowMinX;
	pol.minHilbertY = lowMinY;
	pol.maxHilbertX = lowMaxX;
	pol.maxHilbertY = lowMaxY;
	pol.bufferWidth = lowGridWidth;
	pol.bufferHeight = lowGridHeight;
	// cout << "polygon min/max: (" << pol.minHilbertX << "," << pol.minHilbertY << "),(" << pol.maxHilbertX << "," << pol.maxHilbertY << ")" << endl << endl; 

	//find full cells
	fillAndFinalizeFULLCHECKED(pol, lowM, lowGridWidth, lowGridHeight);
	// cout << "Fill full check (" << lowGridWidth << "x" << lowGridHeight << "): " << endl;
	// printMatrix(lowM, lowGridWidth, lowGridHeight);

	//retrieve the order 16 mapped polygon
	pol = mappedPol;

	//transfer the guaranteed full elements to the high granularity matrix
	transferFullCheckedElements(lowM, lowGridWidth, lowGridHeight, M, highGridWidth, highGridHeight, diffOrder);
	// //print M
	// cout << "After full check transfer (" << pol.bufferWidth << "x" << pol.bufferHeight << "): " << endl;
	// printMatrix(M, pol.bufferWidth, pol.bufferHeight);

	// //clean up uncertain and finalize fulls
	cleanUncertainFinal(M, highGridWidth, highGridHeight, pol, newOrderN);
	// //print
	// cout << "Final matrix: " << endl;
	// printMatrix(M, highGridWidth, highGridHeight);

	//SAVE THE LOW GRANULARITY FULL INTERVALS
	for(auto &p : lowGranularityFullCellsIndices){
		p.first = (p.first * diffOrderPower) + pol.minHilbertX;
		p.second = (p.second * diffOrderPower) + pol.minHilbertY;

		//keep the lowest d from the rectangle defined by (p.first,p.second),(p.first+15,p.second+15)
		d = xy2d(newOrderN, p.first, p.second);
		d = min(d, xy2d(newOrderN, p.first+pow(2,diffOrder)-1, p.second));
		d = min(d, xy2d(newOrderN, p.first, p.second+pow(2,diffOrder)-1));
		d = min(d, xy2d(newOrderN, p.first+pow(2,diffOrder)-1, p.second+pow(2,diffOrder)-1));
		pol.fullCellPackage.intervals.emplace_back(d, d+pow(2,bitsToShift)-1);
	}


	// DON'T FORGET TO DELETE THE MATRIX BEFORE RETURNING!
	for(uint i = 0; i < highGridWidth; i++){
 		delete [] M[i];
	}
	delete [] M;


	//DELETE LOW M
	for(uint i = 0; i < lowGridWidth; i++){
		delete [] lowM[i];
	}
	delete [] lowM;

		
	// // //print partial
	// cout << "PARTIAL" << endl;
	// for(auto &it: pol.partialCellPackage.hilbertCellIDs){
	// 	d2xy(newOrderN, it, x, y);
	// 	cout << "(" << x << "," << y << ")" << endl;		
	// }

	// //print full
	// cout << "FULL" << endl;
	// for(auto &it: pol.fullCellPackage.hilbertCellIDs){
	// 	d2xy(newOrderN, it, x, y);
	// 	cout << "(" << x << "," << y << ")" << endl;			
	// }
}

/* 
*
*
*
*	----- SIMPLE RASTERIZATION -----
*
*
*
*/

void rasterizeSimple(Polygon &pol, Section &sec){
	ID x,y;
	clock_t timer;
	timer = clock();

	//first of all map the polygon's coordinates to this section's hilbert space
	mapPolygonCustom(pol, HILBERT_n, sec);

	// //print mapped polygon
	// for(auto &it: pol.vertices){
	// 	cout << fixed << setprecision(10) << "(" << it.x << "," << it.y << ")" << endl;
	// }
	// cout << endl << endl;

	//allocate enough space for the cells
	pol.partialCellPackage.hilbertCellIDs.reserve(pol.bufferWidth * pol.bufferHeight);
	pol.fullCellPackage.hilbertCellIDs.reserve(pol.bufferWidth * pol.bufferHeight);

	//---------METHOD WITHOUT CLUSTER CLEAN - NEEDS HIGH GRANULARITY AND MANY PARTITIONS TO WORK WELL--------
	/* PARTIAL CELLS */
	//grid traversal algorithm - create partially covered cell matrix
	uint **M = calculatePartialAndUncertain(pol, HILBERT_n);	
	// printMatrix(M, pol.bufferWidth, pol.bufferHeight);


	/* FULL CELLS */
	//flood fill algorithm
	fillAndFinalizeMatrix(pol, M);
	// printMatrix(M, pol.bufferWidth, pol.bufferHeight);

	// DON'T FORGET TO DELETE THE MATRIX BEFORE RETURNING!
	for(size_t i = 0; i < pol.bufferWidth; i++){
		delete M[i];
	}
	delete M;

	//print partial
	// cout << "PARTIAL" << endl;
	// for(auto &it: pol.partialCellPackage.hilbertCellIDs){
	// 	d2xy(HILBERT_n, it, x, y);
	// 	cout << "(" << x << "," << y << ")" << endl;
	// }
	// //print full
	// cout << "FULL" << endl;
	// for(auto &it: pol.fullCellPackage.hilbertCellIDs){
	// 	d2xy(HILBERT_n, it, x, y);
	// 	cout << "(" << x << "," << y << ")" << endl;
	// }
}
