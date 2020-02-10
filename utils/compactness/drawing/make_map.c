static char version_string[]="$Id: make_map.c,v 1.1.1.1 1999/10/15 12:26:21 kise Exp $";
/*
   $Date: 1999/10/15 12:26:21 $
   $Revision: 1.1.1.1 $
   $Author: kise $

   ���顼�ޥåפ���ץ����

   ��: 0x00, ��: 0x01, ���ο�: 0x02 �Ȳ���.
   
   �Ȥ��������Ƥ˰�¸�����ץ����ʤΤ�, ¾�����Ѥ���������
   ��Ƥ��ɬ�ס���

*/

#include <stdio.h>
#include "color.h"
#include "function.h"

void make_map (unsigned short *red,unsigned short *green, unsigned short *blue, int c_rgb )
{
	red[0] = 0xffff;
	red[1] = 0x0000;
	
	green[0] = 0xffff;
	green[1] = 0x0000;
	
	blue[0] = 0xffff;
	blue[1] = 0x0000;

	if(c_rgb == 0x0000ff){
		red[2] = 0x0000;
		green[2] = 0x0000;
		blue[2] = 0xffff;
	}
	else if(c_rgb == 0x00ff00){
		red[2] = 0x0000;
		green[2] = 0xffff;
		blue[2] = 0x0000;
	}
	else{
		red[2] = 0xffff;
		green[2] = 0x0000;
		blue[2] = 0x00000;
	}
}
