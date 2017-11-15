/**
* This file is part of DSO.
* 
* Copyright 2016 Technical University of Munich and Intel.
* Developed by Jakob Engel <engelj at in dot tum dot de>,
* for more information see <http://vision.in.tum.de/dso>.
* If you use this code, please cite the respective publications as
* listed on the above website.
*
* DSO is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* DSO is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with DSO. If not, see <http://www.gnu.org/licenses/>.
*/



#include "util/globalCalib.h"
#include "stdio.h"
#include <iostream>

namespace dso
{
	int wG[PYR_LEVELS], hG[PYR_LEVELS];
	float fxG[PYR_LEVELS], fyG[PYR_LEVELS],
		  cxG[PYR_LEVELS], cyG[PYR_LEVELS];

	float fxiG[PYR_LEVELS], fyiG[PYR_LEVELS],
		  cxiG[PYR_LEVELS], cyiG[PYR_LEVELS];

	Eigen::Matrix3f KG[PYR_LEVELS], KiG[PYR_LEVELS];


	float wM3G;
	float hM3G;

	void setGlobalCalib(int w, int h,const Eigen::Matrix3f &K)
	{
		int wlvl=w;
		int hlvl=h;
		pyrLevelsUsed=1;
		while(wlvl%2==0 && hlvl%2==0 && wlvl*hlvl > 5000 && pyrLevelsUsed < PYR_LEVELS)
		{
			wlvl /=2;
			hlvl /=2;
			pyrLevelsUsed++;
		}
		printf("using pyramid levels 0 to %d. coarsest resolution: %d x %d!\n",
				pyrLevelsUsed-1, wlvl, hlvl);
		if(wlvl>100 && hlvl > 100)
		{
			printf("\n\n===============WARNING!===================\n "
					"using not enough pyramid levels.\n"
					"Consider scaling to a resolution that is a multiple of a power of 2.\n");
		}
		if(pyrLevelsUsed < 3)
		{
			printf("\n\n===============WARNING!===================\n "
					"I need higher resolution.\n"
					"I will probably segfault.\n");
		}

		wM3G = w-3;
		hM3G = h-3;

		wG[0] = w;
		hG[0] = h;
		KG[0] = K;
		fxG[0] = K(0,0);
		fyG[0] = K(1,1);
		cxG[0] = K(0,2);
		cyG[0] = K(1,2);
		KiG[0] = KG[0].inverse();
		fxiG[0] = KiG[0](0,0);
		fyiG[0] = KiG[0](1,1);
		cxiG[0] = KiG[0](0,2);
		cyiG[0] = KiG[0](1,2);

		for (int level = 1; level < pyrLevelsUsed; ++ level)
		{
			wG[level] = w >> level;
			hG[level] = h >> level;

			fxG[level] = fxG[level-1] * 0.5;
			fyG[level] = fyG[level-1] * 0.5;
			cxG[level] = (cxG[0] + 0.5) / ((int)1<<level) - 0.5;
			cyG[level] = (cyG[0] + 0.5) / ((int)1<<level) - 0.5;

			KG[level]  << fxG[level], 0.0, cxG[level], 0.0, fyG[level], cyG[level], 0.0, 0.0, 1.0;	// synthetic
			KiG[level] = KG[level].inverse();

			fxiG[level] = KiG[level](0,0);
			fyiG[level] = KiG[level](1,1);
			cxiG[level] = KiG[level](0,2);
			cyiG[level] = KiG[level](1,2);
		}
	}


    void print_global_calib() {

        printf("------------------------------------------------\n");

        printf("resolution:");
        for ( auto l = 0; l < pyrLevelsUsed; l++) {
            printf("[%d] (%d, %d) ,", l, wG[l], hG[l]);
        }
        printf("\n");

        printf("fx & fy:");
        for ( auto l = 0; l < pyrLevelsUsed; l++) {
            printf("[%d] (%f, %f) ,", l, fxG[l], fyG[l]);
        }
        printf("\n");

        printf("cx & cy:");
        for ( auto l = 0; l < pyrLevelsUsed; l++) {
            printf("[%d] (%f, %f) ,", l, cxG[l], cyG[l]);
        }
        printf("\n");

        printf("fxi & fyi:");
        for ( auto l = 0; l < pyrLevelsUsed; l++) {
            printf("[%d] (%f, %f) ,", l, fxiG[l], fyiG[l]);
        }
        printf("\n");

        printf("cxi & cyi:");
        for ( auto l = 0; l < pyrLevelsUsed; l++) {
            printf("[%d] (%f, %f) ,", l, cxiG[l], cyiG[l]);
        }
        printf("\n");

        printf("wM3G:%f  hM3G:%f\n", wM3G, hM3G);

        printf("........all K.........\n");
        for ( auto l = 0; l < pyrLevelsUsed; l++) {
            printf("level[%d] ---> \n", l);
            std::cout << KG[l] << std::endl;
        }
        printf("........all Ki.........");
        for ( auto l = 0; l < pyrLevelsUsed; l++) {
            printf("level[%d] ---> \n", l);
            std::cout << KiG[l] << std::endl;
        }
    }

}
