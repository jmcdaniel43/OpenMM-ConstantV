#!/usr/bin/perl
#

my@sheets=("graph_n.pdb");

#fill in name for every sheet.
my@name=( "grph" );
#fill in shifts for every sheet, this will shift z-dimension
my@shifts;
@{$shifts[0]}=( 62 , 66 ,  70, 105 , 109, 113 );



#count sheet number;
$sheetnumber=1;
# Loop over sheets and shifts.
for(my$i=0;$i<@sheets;$i++)
{
$ifile=$sheets[$i];
open ifile, $ifile, or die;

# get graphene sheet
$_=<ifile>;
$header=$_;
@store=();
while(<ifile>)
{ if (/END/){last}
 push(@store,$_) }
close(ifile);

# now print this sheet for every shift.
for(my$j=0;$j<@{$shifts[$i]};$j++)
{
# print header if this is the first sheet.
if ( ( $i == 0 ) and ( $j == 0 ) )
{ print $header }

for(my$k=0;$k<@store;$k++)
{
$_=$store[$k];
@array=split;
$z=$array[7]+$shifts[$i][$j];
$e=$array[2];
$e =~ s/[0-9]//g;
printf ("%6s%5d%5s%5s%1s%4d%4s%8.3f%8.3f%8.3f%6.2f%6.2f%12s\n",$array[0], $array[1], $array[2], $name[$i],"X", $sheetnumber,"    ", $array[5], $array[6], $z, 1.0, 0.0, $e );
}
$sheetnumber++;
}
}
print"END\n";
