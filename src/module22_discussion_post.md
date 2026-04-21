# Module 22 - Round 11 Submission + Clustering Reflection Draft

## Part 1: Query submission (portal format)

function_1: 0.000092-0.991177  
function_2: 0.718693-0.783068  
function_3: 0.206391-0.465088-0.309024  
function_4: 0.423371-0.098478-0.466326-0.439897  
function_5: 0.699431-0.999999-0.999999-0.999999  
function_6: 0.729477-0.423161-0.703560-0.836322-0.200699  
function_7: 0.848259-0.000000-0.999999-0.668875-0.000000-0.999999  
function_8: 0.349519-0.265082-0.454653-0.448293-0.440973-0.456735-0.448389-0.621001

## Part 2: Reflection (Discussion board post)

1. How have patterns in your past queries influenced your latest choices?

Earlier rounds showed that some functions repeatedly reward similar local regions rather than completely new areas each time. In this round, I used that pattern through a clustering lens. Functions 1 and 5 still looked boundary-sensitive, so I stayed near strong edge clusters. Functions 2, 3, and 6 looked more stable around recurring local neighborhoods, so I followed centroid-style refinement there. Functions 4 and 7 looked less settled, so I used bridge probes between nearby groups instead of committing immediately to one extreme.

2. Have you identified any ‘clusters’ or recurring regions in your search space that seem promising?

Yes. In the lower-dimensional functions, some promising regions now look like repeated neighborhoods rather than isolated lucky points. Function 4 looks more like a basin with nearby sub-clusters, while Function 5 still behaves like a ridge shaped by a few dominant dimensions near the upper boundary. In the higher-dimensional functions, the clusters are broader and less dense, so separation between groups matters more than raw concentration.

3. Which strategies or parameter choices have proven less effective, and how are you adjusting for them?

The less effective choices were aggressive edge-chasing, relying too heavily on one surrogate family, and treating every uncertain region as equally valuable. I adjusted by tuning the surrogate family first, then using cluster structure to decide whether to follow a centroid trend, tighten a cluster boundary, or probe the gap between clusters. That reduces the chance of overvaluing points that only look attractive because they are novel.

4. In what ways do your refinements parallel how clustering algorithms separate meaningful patterns from noise?

The parallel is that I am now treating repeated local structure as stronger evidence than isolated points. A cluster suggests signal because several observations support the same region, while a lone point might be noise or an unstable pattern. At the same time, I still inspect the space between clusters, because that is often where the useful decision boundary sits. So the refinement is not only “pick the biggest cluster,” but “use clusters to decide whether to exploit, tighten, or bridge.”

5. If your query results were plotted, what trends or groupings might appear? How could these inform your next iteration?

I would expect Functions 1 to 3 to show tighter groups near historically strong regions, with some remaining edge emphasis. Function 4 would likely show a basin with a couple of neighboring pockets. Function 5 might show a stretched ridge where only some dimensions change meaningfully. Functions 6 to 8 would probably show broader, fuzzier clouds with larger distances between useful observations. That visual pattern would help me decide whether the next round should stay within an existing cluster, test its boundary, or move between clusters when the separation is still unclear.
