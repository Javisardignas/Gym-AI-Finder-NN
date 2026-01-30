import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'dart:async';
import 'testing_page.dart';

void main() {
  runApp(const GymAIApp());
}

class GymAIApp extends StatefulWidget {
  const GymAIApp({Key? key}) : super(key: key);

  @override
  State<GymAIApp> createState() => _GymAIAppState();
}

class _GymAIAppState extends State<GymAIApp> {
  bool _isDarkMode = false;

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'GYM AI FINDER',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        primaryColor: const Color(0xFF8B0000),
        useMaterial3: true,
        brightness: Brightness.light,
      ),
      darkTheme: ThemeData(
        primaryColor: const Color(0xFF8B0000),
        useMaterial3: true,
        brightness: Brightness.dark,
      ),
      themeMode: _isDarkMode ? ThemeMode.dark : ThemeMode.light,
      home: SearchPage(
        isDarkMode: _isDarkMode,
        onThemeChanged: (isDark) {
          setState(() => _isDarkMode = isDark);
        },
      ),
    );
  }
}

class SearchPage extends StatefulWidget {
  final bool isDarkMode;
  final Function(bool) onThemeChanged;

  const SearchPage({
    Key? key,
    required this.isDarkMode,
    required this.onThemeChanged,
  }) : super(key: key);

  @override
  State<SearchPage> createState() => _SearchPageState();
}

class _SearchPageState extends State<SearchPage>
    with SingleTickerProviderStateMixin {
  final TextEditingController _controller = TextEditingController();
  List<Map<String, dynamic>> _results = [];
  List<String> _searchHistory = [];
  bool _isLoading = false;
  late AnimationController _animationController;
  late Animation<double> _fadeAnimation;
  late Animation<Offset> _slideAnimation;

  @override
  void initState() {
    super.initState();
    _animationController = AnimationController(
      duration: const Duration(milliseconds: 800),
      vsync: this,
    );
    _fadeAnimation = Tween<double>(begin: 0.0, end: 1.0).animate(
      CurvedAnimation(parent: _animationController, curve: Curves.easeIn),
    );
    _slideAnimation =
        Tween<Offset>(begin: const Offset(0, 0.5), end: Offset.zero).animate(
      CurvedAnimation(parent: _animationController, curve: Curves.easeOut),
    );
  }

  Future<void> _search() async {
    if (_controller.text.trim().isEmpty) return;

    setState(() => _isLoading = true);

    try {
      final response = await http
          .get(Uri.parse(
              'http://localhost:5000/search?q=${Uri.encodeComponent(_controller.text)}'))
          .timeout(const Duration(seconds: 10));

      if (response.statusCode == 200) {
        final json = jsonDecode(response.body);
        if (json['success'] == true) {
          setState(() {
            _results = List<Map<String, dynamic>>.from(json['results']);
            // Add to history
            if (!_searchHistory.contains(_controller.text)) {
              _searchHistory.insert(0, _controller.text);
              if (_searchHistory.length > 10) {
                _searchHistory.removeLast();
              }
            }
          });
          _animationController.forward();
        } else {
          setState(() {
            _results = [
              {'name': 'Error: ${json['error']}', 'similarity': 0.0}
            ];
          });
        }
      } else {
        setState(() {
          _results = [
            {'name': 'Server error: ${response.statusCode}', 'similarity': 0.0}
          ];
        });
      }
    } catch (e) {
      setState(() {
        _results = [
          {
            'name': 'Error: $e (Is server running on localhost:5000?)',
            'similarity': 0.0
          }
        ];
      });
    }

    setState(() => _isLoading = false);
  }

  void _clearSearch() {
    setState(() {
      _controller.clear();
      _results = [];
    });
    _animationController.reset();
  }

  @override
  Widget build(BuildContext context) {
    final isDark = widget.isDarkMode;

    return Scaffold(
      body: Container(
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
            colors: isDark
                ? [Colors.grey[900]!, Colors.grey[800]!]
                : [Colors.grey[100]!, Colors.grey[50]!],
          ),
        ),
        child: CustomScrollView(
          slivers: [
            // Custom AppBar with gradient
            SliverAppBar(
              expandedHeight: 140,
              floating: true,
              pinned: true,
              backgroundColor: const Color(0xFF8B0000),
              elevation: 15,
              shadowColor: const Color(0xFF8B0000).withOpacity(0.7),
              actions: [
                Padding(
                  padding: const EdgeInsets.only(right: 8),
                  child: Container(
                    decoration: BoxDecoration(
                      color: Colors.white.withOpacity(0.2),
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: IconButton(
                      icon: const Icon(
                        Icons.science,
                        color: Colors.white,
                        size: 24,
                      ),
                      onPressed: () async {
                        final result = await Navigator.push(
                          context,
                          MaterialPageRoute(
                            builder: (context) => TestingPage(
                              isDarkMode: widget.isDarkMode,
                              onThemeChanged: widget.onThemeChanged,
                            ),
                          ),
                        );

                        if (!mounted) return;

                        if (result is String && result.trim().isNotEmpty) {
                          setState(() {
                            _controller.text = result.trim();
                            _controller.selection = TextSelection.fromPosition(
                              TextPosition(offset: _controller.text.length),
                            );
                            _results = [];
                          });
                          _animationController.reset();
                        }
                      },
                      tooltip: 'Model Testing',
                    ),
                  ),
                ),
                Padding(
                  padding: const EdgeInsets.only(right: 16, top: 8),
                  child: Container(
                    decoration: BoxDecoration(
                      color: Colors.white.withOpacity(0.2),
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: IconButton(
                      icon: Icon(
                        widget.isDarkMode ? Icons.light_mode : Icons.dark_mode,
                        color: Colors.white,
                        size: 24,
                      ),
                      onPressed: () {
                        widget.onThemeChanged(!widget.isDarkMode);
                      },
                      tooltip: widget.isDarkMode
                          ? 'Switch to Light Mode'
                          : 'Switch to Dark Mode',
                    ),
                  ),
                ),
              ],
              flexibleSpace: FlexibleSpaceBar(
                background: Container(
                  decoration: const BoxDecoration(
                    gradient: LinearGradient(
                      begin: Alignment.topLeft,
                      end: Alignment.bottomRight,
                      colors: [Color(0xFF8B0000), Color(0xFFFF0000)],
                    ),
                  ),
                  child: Center(
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        const Text(
                          '💪 GYM AI FINDER',
                          style: TextStyle(
                            color: Colors.white,
                            fontSize: 32,
                            fontWeight: FontWeight.bold,
                            letterSpacing: 1.5,
                          ),
                        ),
                        const SizedBox(height: 8),
                        Container(
                          padding: const EdgeInsets.symmetric(
                            horizontal: 12,
                            vertical: 4,
                          ),
                          decoration: BoxDecoration(
                            color: Colors.white.withOpacity(0.2),
                            borderRadius: BorderRadius.circular(20),
                          ),
                          child: const Text(
                            'AI-Powered Exercise Search',
                            style: TextStyle(
                              color: Colors.white70,
                              fontSize: 12,
                              fontWeight: FontWeight.w500,
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ),
            ),
            SliverToBoxAdapter(
              child: SingleChildScrollView(
                child: Padding(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 20.0,
                    vertical: 30.0,
                  ),
                  child: Column(
                    children: [
                      // Welcome section
                      Container(
                        padding: const EdgeInsets.all(20),
                        decoration: BoxDecoration(
                          gradient: LinearGradient(
                            colors: isDark
                                ? [
                                    Colors.red.shade900.withOpacity(0.3),
                                    Colors.red.shade700.withOpacity(0.2),
                                  ]
                                : [
                                    Colors.red.shade100.withOpacity(0.6),
                                    Colors.red.shade50.withOpacity(0.4),
                                  ],
                            begin: Alignment.topLeft,
                            end: Alignment.bottomRight,
                          ),
                          borderRadius: BorderRadius.circular(16),
                          border: Border.all(
                            color: const Color(0xFFB22222).withOpacity(0.5),
                            width: 2,
                          ),
                        ),
                        child: Column(
                          children: [
                            const Text(
                              'Find Your Perfect Workout',
                              style: TextStyle(
                                fontSize: 22,
                                fontWeight: FontWeight.bold,
                                color: Color(0xFFFF0000),
                              ),
                            ),
                            const SizedBox(height: 12),
                            Text(
                              'Describe an exercise and let the AI find the best match.',
                              textAlign: TextAlign.center,
                              style: TextStyle(
                                fontSize: 14,
                                color: isDark
                                    ? Colors.grey[300]
                                    : Colors.grey[700],
                                height: 1.5,
                              ),
                            ),
                          ],
                        ),
                      ),
                      const SizedBox(height: 30),

                      // Search field with enhanced styling
                      Container(
                        decoration: BoxDecoration(
                          boxShadow: [
                            BoxShadow(
                              color: const Color(0xFFFF0000).withOpacity(0.3),
                              blurRadius: 20,
                              offset: const Offset(0, 4),
                            ),
                          ],
                        ),
                        child: TextField(
                          controller: _controller,
                          decoration: InputDecoration(
                            hintText:
                                'Press the weight up from your chest, then lower it slowly with control...',
                            hintStyle: TextStyle(
                              color:
                                  isDark ? Colors.grey[500] : Colors.grey[600],
                              fontSize: 14,
                            ),
                            border: OutlineInputBorder(
                              borderRadius: BorderRadius.circular(16),
                              borderSide: const BorderSide(
                                color: Color(0xFFB22222),
                                width: 2,
                              ),
                            ),
                            focusedBorder: OutlineInputBorder(
                              borderRadius: BorderRadius.circular(16),
                              borderSide: const BorderSide(
                                color: Color(0xFFFF0000),
                                width: 3,
                              ),
                            ),
                            prefixIcon: const Padding(
                              padding: EdgeInsets.only(left: 16, right: 12),
                              child: Icon(
                                Icons.fitness_center,
                                color: Color(0xFFB22222),
                                size: 28,
                              ),
                            ),
                            suffixIcon: _controller.text.isNotEmpty
                                ? IconButton(
                                    icon: const Icon(
                                      Icons.clear,
                                      color: Color(0xFFB22222),
                                    ),
                                    onPressed: () {
                                      _controller.clear();
                                      setState(() {});
                                    },
                                  )
                                : null,
                            contentPadding: const EdgeInsets.symmetric(
                              horizontal: 20,
                              vertical: 18,
                            ),
                            filled: true,
                            fillColor:
                                isDark ? Colors.grey[900] : Colors.grey[200],
                          ),
                          style: TextStyle(
                            fontSize: 16,
                            color: isDark ? Colors.white : Colors.black87,
                            fontWeight: FontWeight.w500,
                          ),
                          onChanged: (value) {
                            setState(() {});
                          },
                          onSubmitted: (value) {
                            if (value.isNotEmpty) {
                              _search();
                            }
                          },
                        ),
                      ),
                      const SizedBox(height: 24),

                      // Search and Clear buttons
                      Row(
                        children: [
                          Expanded(
                            child: Container(
                              decoration: BoxDecoration(
                                boxShadow: [
                                  BoxShadow(
                                    color: const Color(0xFFFF0000)
                                        .withOpacity(0.4),
                                    blurRadius: 15,
                                    offset: const Offset(0, 4),
                                  ),
                                ],
                              ),
                              child: ElevatedButton.icon(
                                onPressed: _isLoading ? null : _search,
                                style: ElevatedButton.styleFrom(
                                  backgroundColor: const Color(0xFFFF0000),
                                  disabledBackgroundColor:
                                      Colors.grey[600]!.withOpacity(0.5),
                                  padding: const EdgeInsets.symmetric(
                                    vertical: 16,
                                  ),
                                  shape: RoundedRectangleBorder(
                                    borderRadius: BorderRadius.circular(14),
                                  ),
                                  elevation: 8,
                                ),
                                icon: _isLoading
                                    ? const SizedBox(
                                        width: 20,
                                        height: 20,
                                        child: CircularProgressIndicator(
                                          valueColor:
                                              AlwaysStoppedAnimation<Color>(
                                            Colors.white,
                                          ),
                                          strokeWidth: 2.5,
                                        ),
                                      )
                                    : const Icon(Icons.search, size: 22),
                                label: Text(
                                  _isLoading ? 'Searching...' : 'SEARCH',
                                  style: const TextStyle(
                                    color: Colors.white,
                                    fontSize: 16,
                                    fontWeight: FontWeight.bold,
                                    letterSpacing: 0.5,
                                  ),
                                ),
                              ),
                            ),
                          ),
                          const SizedBox(width: 12),
                          Container(
                            decoration: BoxDecoration(
                              boxShadow: [
                                BoxShadow(
                                  color: Colors.grey.withOpacity(0.3),
                                  blurRadius: 10,
                                  offset: const Offset(0, 2),
                                ),
                              ],
                            ),
                            child: ElevatedButton.icon(
                              onPressed: _results.isEmpty ? null : _clearSearch,
                              style: ElevatedButton.styleFrom(
                                backgroundColor: Colors.grey[700],
                                disabledBackgroundColor:
                                    Colors.grey[600]!.withOpacity(0.3),
                                padding: const EdgeInsets.symmetric(
                                  horizontal: 16,
                                  vertical: 16,
                                ),
                                shape: RoundedRectangleBorder(
                                  borderRadius: BorderRadius.circular(14),
                                ),
                              ),
                              icon: const Icon(Icons.close, size: 20),
                              label: const Text(
                                'CLEAR',
                                style: TextStyle(
                                  color: Colors.white,
                                  fontSize: 14,
                                  fontWeight: FontWeight.bold,
                                ),
                              ),
                            ),
                          ),
                        ],
                      ),
                      const SizedBox(height: 40),

                      // Stats section
                      if (_results.isNotEmpty || _searchHistory.isNotEmpty)
                        Container(
                          padding: const EdgeInsets.all(16),
                          decoration: BoxDecoration(
                            gradient: LinearGradient(
                              colors: isDark
                                  ? [
                                      Colors.red.shade900.withOpacity(0.2),
                                      Colors.red.shade700.withOpacity(0.1),
                                    ]
                                  : [
                                      Colors.red.shade100.withOpacity(0.3),
                                      Colors.red.shade50.withOpacity(0.2),
                                    ],
                            ),
                            borderRadius: BorderRadius.circular(12),
                            border: Border.all(
                              color: const Color(0xFFB22222).withOpacity(0.3),
                            ),
                          ),
                          child: Row(
                            mainAxisAlignment: MainAxisAlignment.spaceAround,
                            children: [
                              _buildStatCard(
                                '🔍',
                                _results.length.toString(),
                                'Results',
                                isDark,
                              ),
                              _buildStatCard(
                                '⏱️',
                                _searchHistory.length.toString(),
                                'History',
                                isDark,
                              ),
                              _buildStatCard(
                                '⚡',
                                '${(_results.isNotEmpty ? (_results[0]['similarity'] * 100).toStringAsFixed(0) : '0')}%',
                                'Match',
                                isDark,
                              ),
                            ],
                          ),
                        ),
                      const SizedBox(height: 30),

                      // Search results
                      if (_results.isNotEmpty)
                        SlideTransition(
                          position: _slideAnimation,
                          child: FadeTransition(
                            opacity: _fadeAnimation,
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Row(
                                  children: [
                                    const Icon(
                                      Icons.check_circle,
                                      color: Color(0xFFFF0000),
                                      size: 28,
                                    ),
                                    const SizedBox(width: 12),
                                    Text(
                                      'Found ${_results.length} Result${_results.length > 1 ? 's' : ''}',
                                      style: const TextStyle(
                                        fontSize: 22,
                                        fontWeight: FontWeight.bold,
                                        color: Color(0xFFFF0000),
                                      ),
                                    ),
                                  ],
                                ),
                                const SizedBox(height: 20),
                                ..._results.asMap().entries.map((entry) {
                                  final index = entry.key;
                                  final result = entry.value;
                                  return Padding(
                                    padding:
                                        const EdgeInsets.only(bottom: 16.0),
                                    child: ScaleTransition(
                                      scale: Tween<double>(begin: 0.8, end: 1.0)
                                          .animate(
                                        CurvedAnimation(
                                          parent: _animationController,
                                          curve: Interval(
                                            index * 0.1,
                                            (index * 0.1) + 0.3,
                                          ),
                                        ),
                                      ),
                                      child: _buildExerciseCard(result, isDark),
                                    ),
                                  );
                                }).toList(),
                              ],
                            ),
                          ),
                        ),

                      // No results message
                      if (_results.isEmpty && !_isLoading)
                        Padding(
                          padding: const EdgeInsets.symmetric(vertical: 40.0),
                          child: Column(
                            children: [
                              Container(
                                padding: const EdgeInsets.all(30),
                                decoration: BoxDecoration(
                                  color: Colors.red.shade900.withOpacity(0.1),
                                  shape: BoxShape.circle,
                                  border: Border.all(
                                    color: const Color(0xFFB22222)
                                        .withOpacity(0.3),
                                    width: 2,
                                  ),
                                ),
                                child: const Icon(
                                  Icons.search_off,
                                  size: 80,
                                  color: Color(0xFFB22222),
                                ),
                              ),
                              const SizedBox(height: 20),
                              Text(
                                'Start searching for exercises',
                                style: const TextStyle(
                                  fontSize: 18,
                                  fontWeight: FontWeight.w600,
                                  color: Color(0xFFFF0000),
                                ),
                              ),
                              const SizedBox(height: 8),
                              Text(
                                'Type in an exercise name or muscle group above',
                                style: TextStyle(
                                  fontSize: 14,
                                  color: isDark
                                      ? Colors.grey[400]
                                      : Colors.grey[600],
                                ),
                              ),
                            ],
                          ),
                        ),

                      const SizedBox(height: 30),

                      // Search history with animations
                      if (_searchHistory.isNotEmpty)
                        Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Padding(
                              padding: const EdgeInsets.only(bottom: 16.0),
                              child: Row(
                                children: [
                                  Container(
                                    padding: const EdgeInsets.all(8),
                                    decoration: BoxDecoration(
                                      color: const Color(0xFFB22222)
                                          .withOpacity(0.2),
                                      borderRadius: BorderRadius.circular(8),
                                    ),
                                    child: const Icon(
                                      Icons.history,
                                      color: Color(0xFFB22222),
                                      size: 24,
                                    ),
                                  ),
                                  const SizedBox(width: 12),
                                  const Text(
                                    'Recent Searches',
                                    style: TextStyle(
                                      fontSize: 20,
                                      fontWeight: FontWeight.bold,
                                      color: Color(0xFFFF0000),
                                    ),
                                  ),
                                ],
                              ),
                            ),
                            Wrap(
                              spacing: 10,
                              runSpacing: 10,
                              children: _searchHistory.map((history) {
                                return GestureDetector(
                                  onTap: () {
                                    _controller.text = history;
                                    _animationController.reset();
                                    _search();
                                  },
                                  child: Container(
                                    padding: const EdgeInsets.symmetric(
                                      horizontal: 16,
                                      vertical: 10,
                                    ),
                                    decoration: BoxDecoration(
                                      gradient: LinearGradient(
                                        colors: [
                                          const Color(0xFFB22222),
                                          const Color(0xFFFF0000),
                                        ],
                                      ),
                                      borderRadius: BorderRadius.circular(20),
                                      boxShadow: [
                                        BoxShadow(
                                          color: const Color(0xFFFF0000)
                                              .withOpacity(0.3),
                                          blurRadius: 8,
                                          offset: const Offset(0, 2),
                                        ),
                                      ],
                                    ),
                                    child: Row(
                                      mainAxisSize: MainAxisSize.min,
                                      children: [
                                        Text(
                                          history,
                                          style: const TextStyle(
                                            color: Colors.white,
                                            fontWeight: FontWeight.w600,
                                            fontSize: 14,
                                          ),
                                        ),
                                        const SizedBox(width: 8),
                                        GestureDetector(
                                          onTap: () {
                                            setState(() =>
                                                _searchHistory.remove(history));
                                          },
                                          child: const Icon(
                                            Icons.close,
                                            color: Colors.white,
                                            size: 16,
                                          ),
                                        ),
                                      ],
                                    ),
                                  ),
                                );
                              }).toList(),
                            ),
                          ],
                        ),

                      const SizedBox(height: 40),
                    ],
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildStatCard(String icon, String value, String label, bool isDark) {
    return Column(
      children: [
        Text(
          icon,
          style: const TextStyle(fontSize: 28),
        ),
        const SizedBox(height: 8),
        Text(
          value,
          style: const TextStyle(
            color: Color(0xFFFF0000),
            fontSize: 20,
            fontWeight: FontWeight.bold,
          ),
        ),
        const SizedBox(height: 4),
        Text(
          label,
          style: TextStyle(
            color: isDark ? Colors.grey[400] : Colors.grey[600],
            fontSize: 12,
            fontWeight: FontWeight.w500,
          ),
        ),
      ],
    );
  }

  Widget _buildExerciseCard(Map<String, dynamic> exercise, bool isDark) {
    final similarity = (exercise['similarity'] as num?)?.toDouble() ?? 0.0;
    final displaySimilarity = similarity > 1.0 ? similarity : similarity * 100;

    Color getSimilarityColor() {
      if (displaySimilarity >= 80) return const Color(0xFF4CAF50);
      if (displaySimilarity >= 60) return const Color(0xFFFFC107);
      return const Color(0xFFFF5252);
    }

    return Container(
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(16),
        boxShadow: [
          BoxShadow(
            color: getSimilarityColor().withOpacity(0.3),
            blurRadius: 12,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Card(
        elevation: 0,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(16),
          side: BorderSide(
            color: getSimilarityColor().withOpacity(0.6),
            width: 2,
          ),
        ),
        color: isDark ? Colors.grey[900] : Colors.grey[100],
        child: Container(
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(16),
            gradient: LinearGradient(
              colors: isDark
                  ? [Colors.grey[900]!, Colors.grey[850]!]
                  : [Colors.grey[100]!, Colors.grey[50]!],
              begin: Alignment.topLeft,
              end: Alignment.bottomRight,
            ),
          ),
          padding: const EdgeInsets.all(20),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // Header with name and score
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          exercise['name'],
                          style: const TextStyle(
                            fontSize: 20,
                            fontWeight: FontWeight.bold,
                            color: Color(0xFFFF0000),
                          ),
                          maxLines: 2,
                          overflow: TextOverflow.ellipsis,
                        ),
                      ],
                    ),
                  ),
                  Container(
                    padding: const EdgeInsets.symmetric(
                      horizontal: 16,
                      vertical: 10,
                    ),
                    decoration: BoxDecoration(
                      gradient: LinearGradient(
                        colors: [
                          getSimilarityColor(),
                          getSimilarityColor().withOpacity(0.8),
                        ],
                      ),
                      borderRadius: BorderRadius.circular(12),
                      boxShadow: [
                        BoxShadow(
                          color: getSimilarityColor().withOpacity(0.4),
                          blurRadius: 8,
                        ),
                      ],
                    ),
                    child: Column(
                      children: [
                        Text(
                          '${displaySimilarity.toStringAsFixed(0)}%',
                          style: const TextStyle(
                            color: Colors.white,
                            fontWeight: FontWeight.bold,
                            fontSize: 20,
                          ),
                        ),
                        const Text(
                          'Match',
                          style: TextStyle(
                            color: Colors.white70,
                            fontSize: 11,
                            fontWeight: FontWeight.w500,
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 16),

              // Similarity bar
              ClipRRect(
                borderRadius: BorderRadius.circular(8),
                child: LinearProgressIndicator(
                  value: displaySimilarity / 100,
                  minHeight: 8,
                  backgroundColor: isDark ? Colors.grey[800] : Colors.grey[300],
                  valueColor: AlwaysStoppedAnimation<Color>(
                    getSimilarityColor(),
                  ),
                ),
              ),
              const SizedBox(height: 12),

              // Description section if available
              if (exercise.containsKey('description') &&
                  exercise['description'] != null &&
                  (exercise['description'] as String).isNotEmpty)
                Padding(
                  padding: const EdgeInsets.only(bottom: 16.0),
                  child: Container(
                    padding: const EdgeInsets.all(12),
                    decoration: BoxDecoration(
                      color: isDark
                          ? Colors.grey[800]?.withOpacity(0.5)
                          : Colors.grey[200]?.withOpacity(0.5),
                      borderRadius: BorderRadius.circular(12),
                      border: Border.all(
                        color: getSimilarityColor().withOpacity(0.3),
                        width: 1,
                      ),
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          'Description',
                          style: TextStyle(
                            fontSize: 12,
                            fontWeight: FontWeight.bold,
                            color: getSimilarityColor(),
                          ),
                        ),
                        const SizedBox(height: 8),
                        Text(
                          exercise['description'] ?? '',
                          style: TextStyle(
                            fontSize: 13,
                            color: isDark ? Colors.grey[300] : Colors.grey[700],
                            height: 1.4,
                          ),
                          maxLines: 3,
                          overflow: TextOverflow.ellipsis,
                        ),
                      ],
                    ),
                  ),
                ),

              // Details row
              Row(
                children: [
                  Icon(
                    Icons.star_rate_rounded,
                    color: getSimilarityColor(),
                    size: 18,
                  ),
                  const SizedBox(width: 8),
                  Expanded(
                    child: Text(
                      'Confidence: ${displaySimilarity.toStringAsFixed(1)}%',
                      style: TextStyle(
                        fontSize: 14,
                        color: isDark ? Colors.grey[400] : Colors.grey[600],
                        fontWeight: FontWeight.w500,
                      ),
                    ),
                  ),
                  Container(
                    padding: const EdgeInsets.symmetric(
                      horizontal: 8,
                      vertical: 4,
                    ),
                    decoration: BoxDecoration(
                      color: getSimilarityColor().withOpacity(0.2),
                      borderRadius: BorderRadius.circular(6),
                    ),
                    child: Text(
                      displaySimilarity >= 80
                          ? 'Excellent'
                          : displaySimilarity >= 60
                              ? 'Good'
                              : 'Fair',
                      style: TextStyle(
                        color: getSimilarityColor(),
                        fontSize: 12,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }

  @override
  void dispose() {
    _controller.dispose();
    _animationController.dispose();
    super.dispose();
  }
}
