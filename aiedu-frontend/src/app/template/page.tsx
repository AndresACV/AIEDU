import React from 'react';
import '../../styles/globals.css';

import Navbar from '../../components/Navbar/index';
import Footer from '../../components/Footer/Footer';

import Banner from '../../components/Banner/index';
import Companies from '../../components/Companies/Companies';
import Courses from '../../components/Courses/index';
import Mentor from '../../components/Mentor/index';
import Testimonials from '../../components/Testimonials/index';
import Newsletter from '../../components/Newsletter/Newsletter';

export default function HomePage() {
  return (
    <>
      <Navbar />
      <main>
        <Banner />
        <Companies />
        <Courses />
        <Mentor />
        <Testimonials />
        <Newsletter />
      </main>
      <Footer />
    </>
  );
}
