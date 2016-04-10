import sys
import os
if __name__ == '__main__':
    pkg_dir = os.path.split(os.path.abspath(__file__))[0]
    parent_dir, pkg_name = os.path.split(pkg_dir)
    is_pygame_pkg = (pkg_name == 'tests' and
                     os.path.split(parent_dir)[1] == 'pygame')
    if not is_pygame_pkg:
        sys.path.insert(0, parent_dir)
else:
    is_pygame_pkg = __name__.startswith('pygame.tests.')

if is_pygame_pkg:
    from pygame.tests.test_utils import test_not_implemented, unittest, \
                                        geterror
else:
    from test.test_utils import test_not_implemented, unittest, geterror

import pygame
try:
    import pygame.freetype as ft
except ImportError:
    ft = None
from pygame.compat import as_unicode, bytes_, unichr_, unicode_


FONTDIR = os.path.join(os.path.dirname (os.path.abspath (__file__)),
                       'fixtures', 'fonts')

def nullfont():
    """return an uninitialized font instance"""
    return ft.Font.__new__(ft.Font)

class FreeTypeFontTest(unittest.TestCase):

    _fixed_path = os.path.join(FONTDIR, 'test_fixed.otf')
    _sans_path = os.path.join(FONTDIR, 'test_sans.ttf')
    _TEST_FONTS = {}

    def setUp(self):
        ft.init()

        if 'fixed' not in self._TEST_FONTS:
            # Inconsolata is an open-source font designed by Raph Levien
            # Licensed under the Open Font License
            # http://www.levien.com/type/myfonts/inconsolata.html
            self._TEST_FONTS['fixed'] = ft.Font(self._fixed_path)

        if 'sans' not in self._TEST_FONTS:
            # Liberation Sans is an open-source font designed by Steve Matteson
            # Licensed under the GNU GPL
            # https://fedorahosted.org/liberation-fonts/
            self._TEST_FONTS['sans'] = ft.Font(self._sans_path)

    def test_freetype_defaultfont(self):
        font = ft.Font(None)
        self.assertEqual(font.name, "FreeSans")

    def test_freetype_Font_init(self):

        self.assertRaises(IOError, ft.Font, os.path.join (FONTDIR, 'nonexistant.ttf'))

        f = self._TEST_FONTS['sans']
        self.assertTrue(isinstance(f, ft.Font))

        f = self._TEST_FONTS['fixed']
        self.assertTrue(isinstance(f, ft.Font))

        f = ft.Font(None, ptsize=24)
        self.assert_(f.height > 0)
        self.assertRaises(IOError, f.__init__,
                          os.path.join(FONTDIR, 'nonexistant.ttf'))
        self.assertRaises(RuntimeError, f.get_rect, 'a', ptsize=24)
        
        # Test attribute preservation during reinitalization
        f = ft.Font(self._sans_path, ptsize=24, ucs4=True)
        self.assertEqual(f.name, 'Liberation Sans')
        self.assertFalse(f.fixed_width)
        self.assertTrue(f.antialiased)
        self.assertFalse(f.oblique)
        self.assertTrue(f.ucs4)
        f.antialiased = False
        f.oblique = True
        f.__init__(self._fixed_path)
        self.assertEqual(f.name, 'Inconsolata')
        ##self.assertTrue(f.fixed_width)
        self.assertFalse(f.fixed_width)  # need a properly marked Mono font
        self.assertFalse(f.antialiased)
        self.assertTrue(f.oblique)
        self.assertTrue(f.ucs4)

    def test_freetype_Font_fixed_width(self):

        f = self._TEST_FONTS['sans']
        self.assertFalse(f.fixed_width)

        f = self._TEST_FONTS['fixed']
        ##self.assertTrue(f.fixed_width)
        self.assertFalse(f.fixed_width)  # need a properly marked Mone font

        self.assertRaises(RuntimeError, lambda : nullfont().fixed_width)

    def test_freetype_Font_get_metrics(self):

        font = self._TEST_FONTS['sans']

        metrics = font.get_metrics('ABCD', ptsize=24)
        self.assertEqual(len(metrics), len('ABCD'))
        self.assertTrue(isinstance(metrics, list))

        for metrics_tuple in metrics:
            self.assertTrue(isinstance(metrics_tuple, tuple), metrics_tuple)
            self.assertEqual(len(metrics_tuple), 6)
            for m in metrics_tuple[:4]:
                self.assertTrue(isinstance(m, int))
            for m in metrics_tuple[4:]:
                self.assertTrue(isinstance(m, float))

        # test for empty string
        metrics = font.get_metrics('', ptsize=24)
        self.assertEqual(metrics, [])

        # test for invalid string
        self.assertRaises(TypeError, font.get_metrics, 24, 24)

        # raises exception when uninitalized
        self.assertRaises(RuntimeError, nullfont().get_metrics,
                          'a', ptsize=24)

    def test_freetype_Font_get_rect(self):

        font = self._TEST_FONTS['sans']

        def test_rect(r):
            self.assertTrue(isinstance(r, pygame.Rect))

        rect_default = font.get_rect("ABCDabcd", ptsize=24)
        test_rect(rect_default)
        self.assertTrue(rect_default.size > (0, 0))
        self.assertTrue(rect_default.width > rect_default.height)

        rect_bigger = font.get_rect("ABCDabcd", ptsize=32)
        test_rect(rect_bigger)
        self.assertTrue(rect_bigger.size > rect_default.size)

        rect_bolden = font.get_rect("ABCDabcd", ptsize=24, style=ft.STYLE_BOLD)
        test_rect(rect_bolden)
        self.assertTrue(rect_bolden.size > rect_default.size)

        font.vertical = True
        rect_vert = font.get_rect("ABCDabcd", ptsize=24)
        test_rect(rect_vert)
        self.assertTrue(rect_vert.width < rect_vert.height)
        font.vertical = False

        rect_oblique = font.get_rect("ABCDabcd", ptsize=24, style=ft.STYLE_OBLIQUE)
        test_rect(rect_oblique)
        self.assertTrue(rect_oblique.width > rect_default.width)
        self.assertTrue(rect_oblique.height == rect_default.height)

        rect_under = font.get_rect("ABCDabcd", ptsize=24, style=ft.STYLE_UNDERLINE)
        test_rect(rect_under)
        self.assertTrue(rect_under.width == rect_default.width)
        self.assertTrue(rect_under.height > rect_default.height)

#        size_utf32 = font.get_size(as_unicode(r'\U000130A7'), ptsize=24)
#        size_utf16 = font.get_size(as_unicode(r'\uD80C\uDCA7'), ptsize=24)
#        self.assertEqual(size_utf16[0], size_utf32[0]);
#        font.utf16_surrogates = False
#        try:
#            size_utf16 = font.get_size(as_unicode(r'\uD80C\uDCA7'), ptsize=24)
#        finally:
#            font.utf16_surrogates = True
#        self.assertNotEqual(size_utf16[0], size_utf32[0]);
        
        self.assertRaises(RuntimeError,
                          nullfont().get_rect, 'a', ptsize=24)

    def test_freetype_Font_height(self):

        f = self._TEST_FONTS['sans']
        self.assertEqual(f.height, 2355)

        f = self._TEST_FONTS['fixed']
        self.assertEqual(f.height, 1100)

        self.assertRaises(RuntimeError, lambda : nullfont().height)
        

    def test_freetype_Font_name(self):

        f = self._TEST_FONTS['sans']
        self.assertEqual(f.name, 'Liberation Sans')

        f = self._TEST_FONTS['fixed']
        self.assertEqual(f.name, 'Inconsolata')

        nf = nullfont()
        self.assertEqual(nf.name, repr(nf))

    def test_freetype_Font_render(self):

        font = self._TEST_FONTS['sans']

        surf = pygame.Surface((800, 600))
        color = pygame.Color(0, 0, 0)

        # make sure we always have a valid fg color
        self.assertRaises(TypeError, font.render, None, 'FoobarBaz')
        self.assertRaises(TypeError, font.render, None, 'FoobarBaz', None)

        # render to new surface
        rend = font.render(None, 'FoobarBaz', pygame.Color(0, 0, 0), None, ptsize=24)
        self.assertTrue(isinstance(rend, tuple))
        self.assertEqual(len(rend), 2)
        self.assertTrue(isinstance(rend[0], pygame.Surface))
        self.assertTrue(isinstance(rend[1], pygame.Rect))
        self.assertEqual(rend[0].get_rect().size, rend[1].size)
        s, r = font.render(None, '', pygame.Color(0, 0, 0), None, ptsize=24)
        self.assertEqual(r.width, 1)
        self.assertEqual(r.height, font.get_sized_height(24))
        self.assertEqual(s.get_size(), r.size)
        self.assertEqual(s.get_bitsize(), 32)

        # render to existing surface
        refcount = sys.getrefcount(surf);
        rend = font.render((surf, 32, 32), 'FoobarBaz', color, None, ptsize=24)
        self.assertEqual(sys.getrefcount(surf), refcount + 1)
        self.assertTrue(isinstance(rend, tuple))
        self.assertEqual(len(rend), 2)
        rsurf, rrect = rend
        self.assertTrue(rsurf is surf)
        self.assertTrue(isinstance(rrect, pygame.Rect))
        self.assertEqual(rrect.top, rrect.height)
##        self.assertEqual(rrect.left, something or other)
        rcopy = rrect.copy()
        rcopy.topleft = (32, 32)
        self.assertTrue(rsurf.get_rect().contains(rcopy))
        
        rect = pygame.Rect(20, 20, 2, 2)
        rend = font.render((surf, rect), 'FoobarBax', color, None, ptsize=24)
        self.assertEqual(rend[1].top, rend[1].height)
        self.assertNotEqual(rend[1].size, rect.size)
        rend = font.render((surf, 20.1, 18.9), 'FoobarBax',
                           color, None, ptsize=24)
##        self.assertEqual(tuple(rend[1].topleft), (20, 18))

        s, r = font.render((surf, rect), '', color, None, ptsize=24)
        self.assertFalse(r)
        self.assertEqual(r.height, font.get_sized_height(24))
        self.assertTrue(s is surf)

        # invalid dest test
        for dest in [0, (), (surf,), (surf, 'a'), (surf, ()),
                     (surf, (1,)), (surf, ('a', 2)), (surf, (1, 'a')),
                     (surf, (1+2j, 2)), (surf, (1, 1+2j)),
                     (surf, 'a', 2), (surf, 1, 'a'), (surf, 1+2j, 2),
                     (surf, 1, 1+2j), (surf, 1, 2, 3)]: 
            self.assertRaises(TypeError, font.render,
                              dest, 'foobar', color, ptsize=24)

        # misc parameter test
        self.assertRaises(ValueError, font.render, None, 'foobar', color)
        self.assertRaises(TypeError, font.render, None, 'foobar', color, "",
                ptsize=24)
        self.assertRaises(ValueError, font.render, None, 'foobar', color, None,
                style=42, ptsize=24)
        self.assertRaises(TypeError, font.render, None, 'foobar', color, None,
                style=None, ptsize=24)
        self.assertRaises(ValueError, font.render, None, 'foobar', color, None,
                style=97, ptsize=24)

        # valid surrogate pairs
#        rend1 = font.render(None, as_unicode(r'\uD800\uDC00'), color, ptsize=24)
#        rend1 = font.render(None, as_unicode(r'\uDBFF\uDFFF'), color, ptsize=24)
#        rend1 = font.render(None, as_unicode(r'\uD80C\uDCA7'), color, ptsize=24)
#        rend2 = font.render(None, as_unicode(r'\U000130A7'), color, ptsize=24)
#        self.assertEqual(rend1[1], rend2[1])
#        font.utf16_surrogates = False
#        try:
#            rend1 = font.render(None, as_unicode(r'\uD80C\uDCA7'),
#                                color, ptsize=24)
#        finally:
#            font.utf16_surrogates = True
#        self.assertNotEqual(rend1[1], rend2[1])
            
        # malformed surrogate pairs
        self.assertRaises(UnicodeEncodeError, font.render,
                          None, as_unicode(r'\uD80C'), color, ptsize=24)
        self.assertRaises(UnicodeEncodeError, font.render,
                          None, as_unicode(r'\uDCA7'), color, ptsize=24)
        self.assertRaises(UnicodeEncodeError, font.render,
                          None, as_unicode(r'\uD7FF\uDCA7'), color, ptsize=24)
        self.assertRaises(UnicodeEncodeError, font.render,
                          None, as_unicode(r'\uDC00\uDCA7'), color, ptsize=24)
        self.assertRaises(UnicodeEncodeError, font.render,
                          None, as_unicode(r'\uD80C\uDBFF'), color, ptsize=24)
        self.assertRaises(UnicodeEncodeError, font.render,
                          None, as_unicode(r'\uD80C\uE000'), color, ptsize=24)

        # raises exception when uninitalized
        self.assertRaises(RuntimeError, nullfont().render,
                          None, 'a', (0, 0, 0), ptsize=24)

        # *** need more unicode testing to ensure the proper glyphs are rendered

    def test_freetype_Font_render_mono(self):
        font = self._TEST_FONTS['sans']
        color = pygame.Color('black')
        colorkey = pygame.Color('white')
        text = "."

        save_antialiased = font.antialiased
        font.antialiased = False
        try:
            surf, r = font.render(None, text, color, ptsize=24)
            self.assertEqual(surf.get_bitsize(), 8)
            flags = surf.get_flags()
            self.assertTrue(flags & pygame.SRCCOLORKEY)
            self.assertFalse(flags & (pygame.SRCALPHA | pygame.HWSURFACE))
            self.assertEqual(surf.get_colorkey(), colorkey)
            self.assertTrue(surf.get_alpha() is None)

            translucent_color = pygame.Color(*color)
            translucent_color.a = 55
            surf, r = font.render(None, text, translucent_color, ptsize=24)
            self.assertEqual(surf.get_bitsize(), 8)
            flags = surf.get_flags()
            self.assertTrue(flags & (pygame.SRCCOLORKEY | pygame.SRCALPHA))
            self.assertFalse(flags & pygame.HWSURFACE)
            self.assertEqual(surf.get_colorkey(), colorkey)
            self.assertEqual(surf.get_alpha(), translucent_color.a)

            surf, r = font.render(None, text, color, colorkey, ptsize=24)
            self.assertEqual(surf.get_bitsize(), 32)
        finally:
            font.antialiased = save_antialiased

    def test_freetype_Font_render_raw(self):
    
        font = self._TEST_FONTS['sans']
        
        text = "abc"
        size = font.get_rect(text, ptsize=24).size
        rend = font.render_raw(text, ptsize=24)
        self.assertTrue(isinstance(rend, tuple))
        self.assertEqual(len(rend), 2)
        r, s = rend
        self.assertTrue(isinstance(r, bytes_))
        self.assertTrue(isinstance(s, tuple))
        self.assertTrue(len(s), 2)
        w, h = s
        self.assertTrue(isinstance(w, int))
        self.assertTrue(isinstance(w, int))
        self.assertEqual(s, size)
        self.assertEqual(len(r), w * h)
        
        r, (w, h) = font.render_raw('', ptsize=24)
        self.assertEqual(w, 0)
        self.assertEqual(h, font.height)
        self.assertEqual(len(r), 0)
        
        # bug with decenders: this would crash
        rend = font.render_raw('render_raw', ptsize=24)

        # bug with non-printable characters: this would cause a crash
        # because the text length was not adjusted for skipped characters.
        text = unicode_("").join([unichr_(i) for i in range(31, 64)])
        rend = font.render_raw(text, ptsize=10)

    def test_freetype_Font_style(self):

        font = self._TEST_FONTS['sans']

        # make sure STYLE_NORMAL is the default value
        self.assertEqual(ft.STYLE_NORMAL, font.style)

        # make sure we check for style type
        try:    font.style = "None"
        except TypeError: pass
        else:   self.fail("Failed style assignement")

        try:    font.style = None
        except TypeError: pass
        else:   self.fail("Failed style assignement")

        # make sure we only accept valid constants
        try:    font.style = 112
        except ValueError: pass
        else:   self.fail("Failed style assignement")

        # make assure no assignements happened
        self.assertEqual(ft.STYLE_NORMAL, font.style)

        # test assignement
        font.style = ft.STYLE_UNDERLINE
        self.assertEqual(ft.STYLE_UNDERLINE, font.style)

        # test complex styles
        st = (  ft.STYLE_BOLD | ft.STYLE_UNDERLINE |
                ft.STYLE_OBLIQUE )

        font.style = st
        self.assertEqual(st, font.style)

        # revert changes
        font.style = ft.STYLE_NORMAL
        self.assertEqual(ft.STYLE_NORMAL, font.style)

    def test_freetype_Font_resolution(self):
        text = "|"  # Differs in width and height
        resolution = ft.get_default_resolution()
        new_font = ft.Font(self._sans_path, resolution=2 * resolution)
        self.assertEqual(new_font.resolution, 2 * resolution)
        size_normal = self._TEST_FONTS['sans'].get_rect(text, ptsize=24).size
        size_scaled = new_font.get_rect(text, ptsize=24).size
        size_by_2 = size_normal[0] * 2
        self.assertTrue(size_by_2 + 2 >= size_scaled[0] >= size_by_2 - 2,
                        "%i not equal %i" % (size_scaled[1], size_by_2))
        size_by_2 = size_normal[1] * 2
        self.assertTrue(size_by_2 + 2 >= size_scaled[1] >= size_by_2 - 2,
                        "%i not equal %i" % (size_scaled[1], size_by_2))
        new_resolution = resolution + 10
        ft.set_default_resolution(new_resolution)
        try:
            new_font = ft.Font(self._sans_path, resolution=0)
            self.assertEqual(new_font.resolution, new_resolution)
        finally:
            ft.set_default_resolution()

    def test_freetype_Font_path(self):
        self.assertEqual(self._TEST_FONTS['sans'].path, self._sans_path)
        self.assertRaises(AttributeError, getattr, nullfont(), 'path')

    # This Font cache test is conditional on freetype being built by a debug
    # version of Python or with the C macro PGFT_DEBUG_CACHE defined.
    def test_freetype_Font_cache(self):
        glyphs = "abcde"
        glen = len(glyphs)
        other_glyphs = "123"
        oglen = len(other_glyphs)
        uempty = unicode_("")
##        many_glyphs = (uempty.join([unichr_(i) for i in range(32,127)] +
##                                   [unichr_(i) for i in range(161,172)] +
##                                   [unichr_(i) for i in range(174,239)]))
        many_glyphs = uempty.join([unichr_(i) for i in range(32,127)])
        mglen = len(many_glyphs)

        count = 0
        access = 0
        hit = 0
        miss = 0

        f = ft.Font(None, ptsize=24, style=ft.STYLE_NORMAL, vertical=False)
        f.antialiased = True

        # Ensure debug counters are zero
        self.assertEqual(f._debug_cache_stats, (0, 0, 0, 0, 0))
        # Load some basic glyphs
        count = access = miss = glen
        f.render_raw(glyphs)
        self.assertEqual(f._debug_cache_stats, (count, 0, access, hit, miss))
        # Vertical should not affect the cache
        access += glen
        hit += glen
        f.vertical = True
        f.render_raw(glyphs)
        f.vertical = False
        self.assertEqual(f._debug_cache_stats, (count, 0, access, hit, miss))
        # New glyphs will
        count += oglen
        access += oglen
        miss += oglen
        f.render_raw(other_glyphs)
        self.assertEqual(f._debug_cache_stats, (count, 0, access, hit, miss))
        # Point size does
        count += glen
        access += glen
        miss += glen
        f.render_raw(glyphs, ptsize=12)
        self.assertEqual(f._debug_cache_stats, (count, 0, access, hit, miss))
        # Underline style does not
        access += oglen
        hit += oglen
        f.underline = True
        f.render_raw(other_glyphs)
        f.underline = False
        self.assertEqual(f._debug_cache_stats, (count, 0, access, hit, miss))
        # Oblique style does
        count += glen
        access += glen
        miss += glen
        f.oblique = True
        f.render_raw(glyphs)
        f.oblique = False
        self.assertEqual(f._debug_cache_stats, (count, 0, access, hit, miss))
        # Bold style does; by this point cache clears can happen
        count += glen
        access += glen
        miss += glen
        f.bold = True
        f.render_raw(glyphs)
        f.bold = False
        ccount, cdelete_count, caccess, chit, cmiss = f._debug_cache_stats
        self.assertEqual((ccount + cdelete_count, caccess, chit, cmiss),
                         (count, access, hit, miss))
        # Rotation does
        count += glen
        access += glen
        miss += glen
        f.render_raw(glyphs, rotation=10)
        ccount, cdelete_count, caccess, chit, cmiss = f._debug_cache_stats
        self.assertEqual((ccount + cdelete_count, caccess, chit, cmiss),
                         (count, access, hit, miss))
        # aliased (mono) glyphs do
        count += oglen
        access += oglen
        miss += oglen
        f.antialiased = False
        f.render_raw(other_glyphs)
        f.antialiased = True
        ccount, cdelete_count, caccess, chit, cmiss = f._debug_cache_stats
        self.assertEqual((ccount + cdelete_count, caccess, chit, cmiss),
                         (count, access, hit, miss))
        # Trigger a cleanup for sure.
        count += 2 * mglen
        access += 2 * mglen
        miss += 2 * mglen
        f.get_metrics(many_glyphs, ptsize=8)
        f.get_metrics(many_glyphs, ptsize=10)
        ccount, cdelete_count, caccess, chit, cmiss = f._debug_cache_stats
        print (ccount, cdelete_count, caccess, chit, cmiss)
        self.assertTrue(ccount < count)
        self.assertEqual((ccount + cdelete_count, caccess, chit, cmiss),
                         (count, access, hit, miss))

    try:
        ft.Font._debug_cache_stats
    except AttributeError:
        del test_freetype_Font_cache

    def test_undefined_character_code(self):
        # To be consistent with pygame.font.Font, undefined codes
        # are rendered as the undefined character, and has metrics
        # of None.
        font = self._TEST_FONTS['sans']

        img, size1 = font.render(None, unichr_(1), (0, 0, 0), ptsize=24)
        img, size0 = font.render(None, "", (0, 0, 0), ptsize=24)
        self.assertTrue(size1.width > size0.width )

        metrics = font.get_metrics(unichr_(1) + unichr_(48), ptsize=24)
        self.assertEqual(len(metrics), 2)
        self.assertTrue(metrics[0] is None)
        self.assertTrue(isinstance(metrics[1], tuple))

class FreeTypeFont(unittest.TestCase):

    def test_resolution(self):
        was_init = ft.was_init()
        if not was_init:
            ft.init()
        try:
            ft.set_default_resolution()
            resolution = ft.get_default_resolution()
            self.assertEqual(resolution, 72)
            new_resolution = resolution + 10
            ft.set_default_resolution(new_resolution)
            self.assertEqual(ft.get_default_resolution(), new_resolution)
            ft.init(resolution=resolution+20)
            self.assertEqual(ft.get_default_resolution(), new_resolution)
        finally:
            ft.set_default_resolution()
            if was_init:
                ft.quit()


if __name__ == '__main__':
    unittest.main()